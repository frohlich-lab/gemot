"""What a PEtab v1 problem tells the neural route: its states, knobs, starts and scales.

Only the data tables are read: measurements, observables, conditions, parameters and, where a
problem ships one, an ``inputs.tsv`` protocol table. The model is never opened. A neural ODE's
whole point here is to ask how much mechanism the DATA carry, so the species, reactions and
parameters of the shipped model would leak the answer being measured -- the same reason the
zero-shot examples ship an empty SBML.

The states are the symbols the observable formulas read that are neither parameters nor
placeholders: ``obs_P = P`` names the state ``P``, fujita's ``scaleFactor * (pAkt + pAkt_S6)``
names two. The formulas themselves are kept, so the staged problem's observation model is the
shipped one and AMICI evaluates it. Condition-table columns are experimental knobs. One that
names a state (``S2`` or ``init_S2``) sets that state's starting amount; any other ``init_*``
column is a dosed species the data never observe, which becomes a named state of its own;
everything else is a clamped input the field reads as a constant. A fixed parameter named
``<state>_init`` or ``init_<state>`` pins that state's start too. These are conventions about
experimental setup and are labelled as such; ``dosed`` lets a caller declare a column dosed where
the convention cannot see it.

Magnitudes are read off the tables as well: the field works in units of each state's and
input's root-mean-square value and of the last measured time, so one width serves a problem
measured in molecules per cell over weeks as well as one in micromolar over minutes.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import petab.v1 as petab
from petab.math import sympify_petab
from petab.v1 import C
from petab.versions import get_major_version

#: Symbols a protocol formula may use besides its declared condition columns.
PROTOCOL_ALLOWED = frozenset({"time", "piecewise", "exp", "log", "log10", "ln", "sqrt", "pow",
                              "abs", "min", "max", "true", "false", "pi"})
_SINGLE = "__single__"
_SYMBOL = re.compile(r"[A-Za-z_]\w*")
_PLACEHOLDER = re.compile(r"^(observable|noise)Parameter\d+_")


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
    #: The yaml the tables were read from; staging reads them again.
    source: Path
    observable_ids: tuple[str, ...]
    #: States the observable formulas read, in order of first appearance.
    observed: tuple[str, ...]
    #: Unobserved states whose starting amount the condition table doses.
    dosed: tuple[str, ...]
    #: Condition columns the field reads as clamped constants.
    inputs: tuple[str, ...]
    timed_inputs: tuple[TimedInput, ...]
    #: Experiment -> every condition column's value there, a parameter read as its nominal.
    condition_values: dict[str, dict[str, float]]
    #: Condition column -> the state whose start it sets.
    start_columns: dict[str, str]
    #: ``(experiment, state)`` a start column actually pins. A column sets a start only where its
    #: cell carries a value: a blank reads as zero like any other missing knob, and zero is a
    #: legitimate start, so the column's mere existence cannot stand in for the cell being filled.
    column_starts: frozenset[tuple[str, str]]
    #: State -> the start a fixed ``<state>_init`` parameter pins in every experiment.
    pinned_starts: dict[str, float]
    #: ``(experiment, state)`` -> the t=0 datum of an identity readout.
    measured_starts: dict[tuple[str, str], float]
    experiment_ids: tuple[str, ...]
    #: Experiment -> its preequilibration condition; empty without preequilibration.
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

    @property
    def state_names(self) -> tuple[str, ...]:
        return (*self.observed, *self.dosed)

    @property
    def field_input_names(self) -> tuple[str, ...]:
        """What the field reads besides its states: clamped columns no protocol consumes, then the
        protocol targets."""
        consumed = {c for item in self.timed_inputs for c in item.columns}
        return tuple((*(n for n in self.inputs if n not in consumed),
                      *(item.target for item in self.timed_inputs)))

    @property
    def uses_preequilibration(self) -> bool:
        return bool(self.preequilibration)

    @property
    def basal_conditions(self) -> tuple[str, ...]:
        """Distinct preequilibration conditions, in order of first use."""
        return tuple(dict.fromkeys(self.preequilibration[e] for e in self.experiment_ids
                                   if e in self.preequilibration))

    def start_is_known(self, experiment: str, state: str) -> bool:
        """Whether the tables pin ``state``'s start in ``experiment`` at no parameter cost: a t=0
        datum, a setup-sheet column or a fixed ``<state>_init``. bruno starts at t=5 and crauste at
        t=4, so this is read off the data rather than configured.

        The column clause is per experiment, not per column: an incomplete start column leaves the
        experiments whose cell is blank unpinned, and counting those as fixed would understate
        ``k``, and an over-generous parameter budget is how a field gets latents the data cannot pay for.
        """
        return (state in self.pinned_starts or (experiment, state) in self.column_starts
                or (experiment, state) in self.measured_starts)

    def counts(self, *, shared_ics: bool) -> dict:
        """The counts the field-sizing rules read, off this problem's own tables. ``n_fixed_ics``
        is per experiment, or per state when starts are shared across experiments."""
        known = [[self.start_is_known(e, s) for s in self.state_names] for e in self.experiment_ids]
        n_fixed = (sum(all(column) for column in zip(*known)) if shared_ics and known
                   else sum(map(sum, known)))
        return {"n_measurements": self.n_measurements, "n_experiments": len(self.experiment_ids),
                "n_modelled_states": len(self.observed), "n_dosed": len(self.dosed),
                "n_inputs": len(self.field_input_names), "n_fixed_ics": int(n_fixed),
                "n_sigma": len(self.noise_parameters)}


def load(problem: Path | str, *, name: str | None = None, dosed: tuple[str, ...] = ()) -> Problem:
    """Read a PEtab v1 problem directory (or its yaml) into a :class:`Problem`.

    ``dosed`` names condition columns that are dosed species although no naming convention says so
    (enzyme_kinetics' ``S`` is substrate added at t=0 and consumed). Raises :class:`ProblemError`
    on a v2 problem or a table this reader cannot make consistent.
    """
    path = Path(problem)
    yaml_path = path if path.is_file() else path / "problem.yaml"
    if get_major_version(str(yaml_path)) != 1:
        raise ProblemError(f"{yaml_path} is not a PEtab v1 problem; this route reads v1 directly")
    pp = petab.Problem.from_yaml(str(yaml_path))
    name = name or (yaml_path.parent.parent.name if yaml_path.parent.name == "problem"
                    else yaml_path.parent.name)
    warnings: list[str] = []

    meas = pp.measurement_df.reset_index(drop=True).copy()
    if meas.empty:
        raise ProblemError(f"{name}: the problem has no measurements")
    for column in (C.SIMULATION_CONDITION_ID, C.PREEQUILIBRATION_CONDITION_ID):
        if column not in meas.columns:
            meas[column] = _SINGLE if column == C.SIMULATION_CONDITION_ID else None
    meas[C.SIMULATION_CONDITION_ID] = (meas[C.SIMULATION_CONDITION_ID].fillna(_SINGLE)
                                       .astype(str).replace("", _SINGLE))
    experiment_ids = tuple(pd.unique(meas[C.SIMULATION_CONDITION_ID]))

    observables = pp.observable_df.reset_index()
    observable_ids = tuple(str(o) for o in observables[C.OBSERVABLE_ID])
    unknown = set(meas[C.OBSERVABLE_ID].astype(str)) - set(observable_ids)
    if unknown:
        raise ProblemError(f"{name}: measurements reference undeclared observables {sorted(unknown)}")
    nominal, estimated = _parameters(pp.parameter_df)
    formulas = {oid: str(f) for oid, f in zip(observable_ids, observables[C.OBSERVABLE_FORMULA])}
    observed: list[str] = []
    for formula in formulas.values():
        for symbol in _symbols(formula):
            if symbol not in nominal and symbol not in estimated and symbol not in observed:
                observed.append(symbol)
    if not observed:
        raise ProblemError(f"{name}: no observable formula reads a state; nothing to model")

    conditions = pp.condition_df
    targets = [str(c) for c in (conditions.columns if conditions is not None else ())
               if str(c) != C.CONDITION_NAME]
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

    preequilibration = _preequilibration(meas, experiment_ids, name)
    # The preequilibration conditions are read too, not just the experiments'. Their input values
    # are the ONLY place a gated input takes its other value -- perelson's `rit` is 1 in its single
    # experiment and 0 only in `preeq_untreated` -- and the distiller needs both to recover any
    # dependence on it at all (`distil.sample`).
    basal = tuple(dict.fromkeys(preequilibration.values()))
    values, given = _condition_values(conditions, targets, (*experiment_ids, *basal), nominal, warnings)
    column_starts = frozenset(
        (experiment, state)
        for target, state in start_columns.items()
        for e, experiment in enumerate(experiment_ids) if given[target][e])
    identity = {oid: f for oid, f in formulas.items() if f.strip() in observed}
    measured_starts: dict[tuple[str, str], float] = {}
    at_start = meas[(meas[C.TIME] == 0.0) & meas[C.OBSERVABLE_ID].isin(identity)]
    for (experiment, oid), rows in at_start.groupby([C.SIMULATION_CONDITION_ID, C.OBSERVABLE_ID]):
        measured_starts[(str(experiment), identity[str(oid)].strip())] = float(
            rows[C.MEASUREMENT].astype(float).mean())

    scales: dict[str, float] = {}
    for state in observed:
        scales[state] = _observed_scale(state, meas, formulas, nominal)
    n_experiments = len(experiment_ids)
    for target, state in start_columns.items():
        if state in dosed_states:
            scales[state] = _rms(values[target][:n_experiments])
    for target in inputs:
        scales[target] = _rms(values[target][:n_experiments])
    finite = meas[C.TIME].astype(float)
    finite = finite[np.isfinite(finite)]
    horizon = float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
    timed = _timed_inputs(yaml_path.parent, inputs, warnings)
    for item in timed:
        scales[item.target] = 1.0

    return Problem(
        name=name, source=yaml_path, observable_ids=observable_ids, observed=tuple(observed),
        dosed=tuple(dosed_states), inputs=tuple(inputs), timed_inputs=timed,
        condition_values={e: {t: float(values[t][i]) for t in targets}
                          for i, e in enumerate((*experiment_ids, *basal))},
        start_columns=start_columns, column_starts=column_starts, pinned_starts=pinned_starts,
        measured_starts=measured_starts,
        experiment_ids=experiment_ids, preequilibration=preequilibration,
        n_measurements=int(len(meas)),
        noise_parameters=_noise_parameters(meas, observables, nominal, estimated),
        observable_parameters=_observable_parameters(pp, estimated),
        scales=scales, horizon=horizon, warnings=tuple(dict.fromkeys(warnings)))


def _symbols(formula: str) -> list[str]:
    """Free symbols of a PEtab formula in order of appearance, placeholders excluded."""
    names = [str(s) for s in sympify_petab(formula).free_symbols if not _PLACEHOLDER.match(str(s))]
    return sorted(names, key=formula.find)


def _parameters(parameter_df) -> tuple[dict, dict]:
    """``{id: nominalValue}`` and ``{id: estimated}``; a row without a value counts as estimated."""
    nominal, estimated = {}, {}
    if parameter_df is None or parameter_df.empty:
        return nominal, estimated
    for pid, row in parameter_df.iterrows():
        value = _as_float(row.get(C.NOMINAL_VALUE))
        flag = row.get(C.ESTIMATE)
        estimated[str(pid)] = value is None or bool(int(float(flag))) if not _blank(flag) else True
        if value is not None:
            nominal[str(pid)] = value
    return nominal, estimated


def _noise_parameters(meas, observables, nominal, estimated) -> tuple[str, ...]:
    """Distinct symbols the noise formulas resolve to that the parameter table estimates: a
    ``noiseFormula`` naming a parameter, or a placeholder the measurements override with one."""
    found: dict[str, None] = {}
    overrides = (meas[C.NOISE_PARAMETERS].astype(str) if C.NOISE_PARAMETERS in meas.columns
                 else pd.Series([""] * len(meas)))
    for _, row in observables.iterrows():
        formula = "" if _blank(row.get(C.NOISE_FORMULA)) else str(row[C.NOISE_FORMULA])
        # Free symbols, not a regex over identifiers: `sigma * exp(-decay * time)` makes `exp` look
        # like a name, and since no parameter row carries it, the estimated-by-default rule below
        # would count it as a sigma -- inflating `n_sigma` and naming a parameter that does not
        # exist. `time` is PEtab's independent variable and reaches free_symbols as an ordinary
        # symbol, so it needs excluding too, for the same reason and with the same consequence.
        # The measurement overrides stay a regex: those cells are raw text, not a formula.
        names = ({str(s) for s in sympify_petab(formula).free_symbols} - {C.TIME}
                 if formula.strip() else set())
        if any(_PLACEHOLDER.match(n) for n in names):
            rows = (meas[C.OBSERVABLE_ID].astype(str) == str(row[C.OBSERVABLE_ID])).to_numpy()
            for cell in overrides[rows]:
                names.update(n for n in str(cell).split(C.PARAMETER_SEPARATOR) if _SYMBOL.fullmatch(n.strip()))
        for symbol in sorted(names):
            if not _PLACEHOLDER.match(symbol) and estimated.get(symbol, symbol not in nominal):
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


def _observed_scale(state, meas, formulas, nominal) -> float:
    """RMS of the measurements reading ``state``, each divided by its observable's gain in the
    state, so the number is the state's own magnitude rather than the readout's."""
    parts = []
    for oid, formula in formulas.items():
        expr = sympify_petab(formula)
        if state not in {str(s) for s in expr.free_symbols}:
            continue
        rows = meas[meas[C.OBSERVABLE_ID].astype(str) == oid]
        cells = (rows[C.OBSERVABLE_PARAMETERS] if C.OBSERVABLE_PARAMETERS in rows.columns
                 else pd.Series([None] * len(rows), index=rows.index))
        gains = {cell: _gain(expr, state, nominal, cell) for cell in cells.astype(str).unique()}
        parts.append(rows[C.MEASUREMENT].to_numpy(dtype=float) / cells.astype(str).map(gains).to_numpy())
    return _rms(np.concatenate(parts)) if parts else 1.0


def _gain(expr, state: str, nominal, cell) -> float:
    """The observable's slope in ``state``: the formula at ``state = 1`` less the formula at
    ``state = 0``, with every parameter at its nominal, every placeholder at the row's override
    and every other state at zero -- so an additive offset does not enter the scale."""
    placeholders = sorted((str(s) for s in expr.free_symbols if _PLACEHOLDER.match(str(s))),
                          key=lambda n: int(re.search(r"\d+", n).group()))
    overrides = [] if _blank(cell) or cell == "None" else str(cell).split(C.PARAMETER_SEPARATOR)
    values = {}
    own = [symbol for symbol in expr.free_symbols if str(symbol) == state]
    for symbol in expr.free_symbols:
        name = str(symbol)
        if name == state:
            continue
        if name in placeholders:
            index = placeholders.index(name)
            override = overrides[index].strip() if index < len(overrides) else ""
            values[symbol] = _as_float(override) if _as_float(override) is not None else nominal.get(override, 1.0)
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


def _condition_values(conditions, targets, experiment_ids, nominal,
                      warnings) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per column, one value per experiment: numeric cells as given, a parameter's name as its
    nominal value; a missing row or cell reads as zero, and says so.

    Also returns, per column, which experiments actually carried a value. Zero is a legitimate
    start, so the values alone cannot say whether a cell was filled in -- and a start column with
    a hole in it pins no start for that experiment, however the column is named.
    """
    values = {target: np.zeros(len(experiment_ids)) for target in targets}
    given = {target: np.zeros(len(experiment_ids), dtype=bool) for target in targets}
    for e, experiment in enumerate(experiment_ids):
        if conditions is None or experiment not in conditions.index:
            if targets:
                warnings.append(f"experiment {experiment!r} has no condition row; knobs read as 0")
            continue
        for target in targets:
            cell = conditions.loc[experiment, target]
            value = _as_float(cell)
            if value is not None:
                given[target][e] = True
            elif str(cell).strip() in nominal:
                value = nominal[str(cell).strip()]
                given[target][e] = True
            else:
                if not _blank(cell):
                    warnings.append(f"{target}={cell!r} in {experiment!r} is not numeric; read as 0")
                value = 0.0
            values[target][e] = value
    return values, given


def _preequilibration(meas, experiment_ids, name) -> dict[str, str]:
    """Experiment -> preequilibration condition, for every experiment or none."""
    found: dict[str, str] = {}
    for experiment, value in zip(meas[C.SIMULATION_CONDITION_ID], meas[C.PREEQUILIBRATION_CONDITION_ID]):
        if not _blank(value):
            found.setdefault(str(experiment), str(value).strip())
    if not found:
        return {}
    missing = [e for e in experiment_ids if e not in found]
    if missing:
        raise ProblemError(f"{name}: experiments {missing} preequilibrate in no condition while others do")
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
            columns = tuple(c.strip() for c in str(row.get("conditionInputs") or "").split(";") if c.strip())
            if not target or not formula:
                continue
            symbols = {s for s in _SYMBOL.findall(formula) if s.lower() not in PROTOCOL_ALLOWED}
            outside = (symbols - set(columns)) | (set(columns) - set(input_names))
            if outside:
                warnings.append(f"input {target!r} REFUSED: it names {sorted(outside)}, which are "
                                "not clamped condition columns of this problem")
                continue
            out.append(TimedInput(target, formula, columns))
    return tuple(out)


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip())


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
