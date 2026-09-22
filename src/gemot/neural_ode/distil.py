"""Distil the trained field into rate laws, and out as Antimony the judge can score.

A neural ODE cannot be judged against ``reference.json``: BIC charges ``ln(n)`` per parameter and
the network has hundreds. A distilled expression has a handful of constants, so it drops into the
same comparison as every other model, and as Antimony it is committed through ``make_changes``
and scored by the existing judge rather than by a second yardstick.

Two choices decide whether a distillation means anything. **Sample where the data are**: the
field is constrained only where the fitted trajectories went, so the samples are drawn from those
trajectories, jittered a little so the regression sees a neighbourhood and can tell one state's
influence from another's, never from a box over state space. **Distil the derivative**: the target
is the network's output at the sample, which makes the dynamical inverse problem an ordinary
regression and shows a distillation error as a poor fit of the FIELD, before any re-integration.

The regression runs in normalised units, the network's own; the emitted model is in data units:
with ``z = y / s`` and a regressed ``dz/dt = g(z, u)``, ``dy/dt = s / horizon * g(y / s, u / r)``,
and every fitted constant of the data-unit expression becomes a named parameter seeded at its
value, shared where one magnitude appears in two equations, integers left as structure. Species
keep their names, so the shipped observables apply unchanged; the condition columns are declared
as parameters the condition table sets; a protocol stays an assignment rule.

PySR is optional: without it the result says so and carries no model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp

from . import stage as S
from . import train as T
from .problem import Problem

#: Saturation as one primitive, priced like the mechanism it is; ``abs`` keeps it finite where the
#: sampled states go negative. Named ``hill`` in ``binary_operators`` and off by default: on four
#: benchmarks it enlarged the search without buying accuracy.
HILL_OPERATOR = "hill(x, k) = x / (k * k + abs(x) + eps(x))"
DEFAULT_BINARY_OPERATORS = ("+", "-", "*", "/")
#: Rows the basal (preequilibration) state contributes to the sample, per experiment. Enough that
#: the regression sees the gated input at both values, few enough not to outweigh the trajectory.
_BASAL_SAMPLE_ROWS = 25
#: What the server reserves for a measurement mapping (``petab_mcp.validation``,
#: ``petab_mcp.reconcile``): a symbol so named must be an assignment rule, never a species.
READOUT_PREFIX = "readout_"


@dataclass(frozen=True)
class DistilConfig:
    """The PySR sweep axis. Defaults chosen for rate laws, not for general regression."""

    n_samples: int = 2000
    #: Relative to each state's spread across the trajectories.
    jitter: float = 0.05
    #: Dense time points per experiment the trajectories are sampled on.
    n_times: int = 200
    niterations: int = 40
    population_size: int = 33
    maxsize: int = 20
    parsimony: float = 0.0032
    binary_operators: tuple[str, ...] = DEFAULT_BINARY_OPERATORS
    unary_operators: tuple[str, ...] = ()
    #: PySR's own accuracy/complexity pick, or the lowest-loss row.
    selection: str = "best"
    #: Searches run at once, one per state; each is then serial.
    workers: int = 1
    #: How one search parallelises, and its thread count (0 lets PySR choose).
    parallelism: str = "multithreading"
    procs: int = 0
    #: The Python holding PySR; this interpreter when unset.
    interpreter: str | None = None
    timeout_seconds: float = 3600.0
    seed: int = 0

    def key(self) -> str:
        return f"pysr_it{self.niterations}_max{self.maxsize}_p{self.parsimony:g}_s{self.seed}"


@dataclass(frozen=True)
class Candidate:
    """One complete system off PySR's fronts: for a complexity budget every state contributes its
    most accurate expression that fits under it, so the family is monotone in complexity and each
    member is coherent."""

    complexity_budget: int
    expressions: tuple[sp.Expr, ...]
    complexities: tuple[int, ...]
    field_r2: tuple[float, ...]
    n_constants: int
    antimony: str


@dataclass
class DistilResult:
    problem_name: str
    config: DistilConfig
    state_names: tuple[str, ...]
    #: The ``selection`` pick per state, in normalised units; ``antimony`` is its data-unit model.
    expressions: tuple[sp.Expr, ...]
    field_r2: tuple[float, ...]
    antimony: str
    n_constants: int
    candidates: tuple[Candidate, ...] = ()
    pysr_available: bool = True
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.problem_name} [{self.config.key()}] {self.n_constants} constants"]
        lines += [f"  d{n}/dt = {e}   (field R^2 = {r:.4f})" for n, e, r in
                  zip(self.state_names, self.expressions, self.field_r2)]
        return "\n".join((*lines, *(f"  warning: {w}" for w in self.warnings)))


class RegressionUnavailable(RuntimeError):
    """The regression could not run at all, as opposed to running and finding little."""


def distil(problem: Problem, result: T.TrainResult, config: DistilConfig | None = None) -> DistilResult:
    """Regress every state's derivative from the trained field and emit the models.

    Without PySR the result comes back with ``pysr_available`` false and zero-expression
    placeholders, so a sweep on a Julia-less machine records why it is empty instead of dying.
    """
    config = config or DistilConfig()
    staged = result.staged
    names = tuple(n for n in staged.states)
    trajectories = T.trajectories(result, n_times=config.n_times)
    features, derivatives, variables = sample(problem, result, config, trajectories)
    placeholders = [f"x{i}" for i in range(len(variables))]
    back = {sp.Symbol(p): sp.Symbol(v) for p, v in zip(placeholders, variables)}
    try:
        outputs = _regress(features, derivatives, placeholders, config)
    except RegressionUnavailable as why:
        return DistilResult(problem.name, config, names, tuple(sp.Integer(0) for _ in names),
                            tuple(float("nan") for _ in names), "", 0, pysr_available=False,
                            warnings=[f"the symbolic regression did not run: {why}"])
    expressions = tuple(sp.sympify(o["selected"]).subs(back) for o in outputs)
    fronts = [[(int(c), sp.sympify(text).subs(back), float(r2)) for c, text, r2 in o["front"]] for o in outputs]
    starts = _starts(result, trajectories)
    emitted = emit(problem, staged, names, expressions, starts)
    return DistilResult(problem.name, config, names, expressions, tuple(float(o["r2"]) for o in outputs),
                        emitted[0], emitted[1], candidates=_candidates(problem, staged, names, fronts, starts))


def sample(problem: Problem, result: T.TrainResult, config: DistilConfig,
           trajectories: dict | None = None) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """``(features, derivatives, variables)`` in the network's normalised units: rows drawn from
    the dense trajectories with jitter, each carrying its experiment's inputs and the protocol
    value at its time; the derivatives are the network's outputs there."""
    staged = result.staged
    trajectories = trajectories or T.trajectories(result, n_times=config.n_times)
    order = T.state_names(result)
    rows, times, experiments = [], [], []
    for experiment, arrays in trajectories.items():
        finite = np.isfinite(arrays["states"]).all(axis=1)
        rows.append(arrays["states"][finite])
        times.append(arrays["times"][finite])
        experiments += [experiment] * int(finite.sum())
        # The BASAL regime as well, where there is one. A gated input takes its other value only
        # in the preequilibration condition -- perelson's `rit` is 1 in its one experiment and 0
        # in `preeq_untreated` -- so sampling the experiments alone leaves that column constant,
        # the regression cannot see the input at all, and the server then rejects every proposal
        # for "the condition table sets 'rit' ... but the model never uses 'rit'". The state there
        # is the trajectory's own t=0, which IS what preequilibration relaxed to, and the field is
        # at rest in it: the rows say dx/dt = 0 at the untreated steady state, which is true.
        basal = problem.preequilibration.get(experiment)
        if basal is not None and finite.any():
            settled = arrays["states"][finite][:1]
            rows.append(np.repeat(settled, _BASAL_SAMPLE_ROWS, axis=0))
            times.append(np.zeros(_BASAL_SAMPLE_ROWS))
            experiments += [basal] * _BASAL_SAMPLE_ROWS
    states = np.concatenate(rows)
    if not states.size:
        raise ValueError("the fitted field produced no finite trajectory to sample")
    times, experiments = np.concatenate(times), np.asarray(experiments)
    rng = np.random.default_rng(config.seed)
    draw = rng.integers(0, len(states), size=int(config.n_samples))
    spread = np.maximum(states.std(axis=0), 1e-9)
    picked = states[draw] + rng.normal(0.0, config.jitter, (len(draw), states.shape[1])) * spread
    by_name = {name: picked[:, i] for i, name in enumerate(order)}
    columns = []
    for name in staged.field_inputs:
        if name in by_name:
            columns.append(by_name[name] / staged.scales[name])
        elif name in problem.inputs:
            columns.append(np.array([problem.condition_values[e][name] for e in experiments[draw]]) / staged.scales[name])
        else:
            columns.append(_protocol_values(problem, name, times[draw], experiments[draw]) / staged.scales[name])
    features = np.stack(columns, axis=1)
    network = result.problem.model.nns[S.NETWORK_ID]
    derivatives = np.asarray(network.forward(features))
    return features, derivatives, staged.field_inputs


def _protocol_values(problem: Problem, target: str, times: np.ndarray, experiments: np.ndarray) -> np.ndarray:
    """A protocol's value at each sample, from its formula over time and its condition columns."""
    from petab.math import sympify_petab  # noqa: PLC0415

    item = next(t for t in problem.timed_inputs if t.target == target)
    expr = sympify_petab(item.formula)
    symbols = [sp.Symbol("time"), *(sp.Symbol(c) for c in item.columns)]
    fn = sp.lambdify(symbols, expr, modules=["numpy"])
    values = np.array([[problem.condition_values[e][c] for c in item.columns] for e in experiments])
    return np.asarray([float(fn(t, *row)) for t, row in zip(times, values)])


def _regress(features, derivatives, placeholders, config: DistilConfig) -> list[dict]:
    """The worker, by file path, in ``config.interpreter``; its stderr tail is the failure's reason."""
    settings = {"niterations": config.niterations, "population_size": config.population_size,
                "maxsize": config.maxsize, "parsimony": config.parsimony,
                "binary_operators": [HILL_OPERATOR if op == "hill" else op for op in config.binary_operators],
                "unary_operators": list(config.unary_operators),
                "seed": config.seed, "selection": config.selection, "placeholders": list(placeholders),
                "workers": config.workers, "parallelism": config.parallelism, "procs": config.procs}
    with tempfile.TemporaryDirectory(prefix="neural_ode_pysr_") as directory:
        root = Path(directory)
        np.savez(root / "in.npz", features=features, derivatives=derivatives)
        (root / "settings.json").write_text(json.dumps({**settings, "output_directory": str(root / "pysr")}))
        command = [config.interpreter or sys.executable, str(Path(__file__).with_name("pysr_worker.py")),
                   str(root / "in.npz"), str(root / "settings.json"), str(root / "out.json")]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired as why:
            raise RegressionUnavailable(f"the PySR worker exceeded {config.timeout_seconds:.0f}s") from why
        if completed.returncode != 0 or not (root / "out.json").is_file():
            tail = " | ".join((completed.stderr or completed.stdout or "").strip().splitlines()[-6:])
            raise RegressionUnavailable(f"the PySR worker exited {completed.returncode}: {tail}")
        return json.loads((root / "out.json").read_text())


def _starts(result: T.TrainResult, trajectories: dict) -> dict[str, float]:
    """Every state's start in the first experiment, as simulated: pinned or fitted alike."""
    first = next(iter(trajectories.values()))["states"][0]
    return {name: float(v) for name, v in zip(T.state_names(result), first)}


def _candidates(problem, staged, names, fronts, starts) -> tuple[Candidate, ...]:
    if not fronts or any(not front for front in fronts):
        return ()
    budgets = sorted({c for front in fronts for c, _, _ in front})
    out, seen = [], set()
    for budget in budgets:
        picks = []
        for front in fronts:
            affordable = [item for item in front if item[0] <= budget]
            if not affordable:
                break
            picks.append(max(affordable, key=lambda item: (item[2], -item[0])))
        if len(picks) != len(fronts):
            continue
        signature = tuple(c for c, _, _ in picks)
        if signature in seen:
            continue
        seen.add(signature)
        expressions = tuple(e for _, e, _ in picks)
        antimony, n_constants = emit(problem, staged, names, expressions, starts)
        out.append(Candidate(budget, expressions, signature, tuple(r for _, _, r in picks), n_constants, antimony))
    return tuple(out)


def emit(problem: Problem, staged: S.Staged, names, expressions, starts: dict[str, float]) -> tuple[str, int]:
    """The system as Antimony in data units, with rate rules: a distilled derivative is not a sum
    of fluxes, so no stoichiometry is invented. Returns the text and its constant count."""
    rendered, constants = _data_units(staged, names, expressions)
    species = [*names, *(["clock"] if "clock" in staged.field_inputs else [])]
    # A `readout_*` symbol an observable references is a MEASUREMENT MAPPING: the server refuses a
    # problem that declares one as a species ("a free constant is not a measurement"), because the
    # benchmark examples blind their observables to those names and their canonical builds define
    # them with assignment rules. The route's states carry the readout names so the shipped
    # observables apply unchanged, so the state is emitted under its own id and the readout becomes
    # the identity rule onto it -- the same model, in the shape the judge accepts.
    emitted = {name: _species_id(name, species, problem) for name in species}
    lines = [f"model {_identifier(problem.name)}_distilled",
             f"  species {', '.join(emitted[name] for name in species)};"]
    for name in species:
        lines.append(f"  {emitted[name]} = {starts.get(name, 0.0):.10g};")
    if problem.inputs:
        lines.append("  // per-condition experimental inputs, set by the condition table")
        lines.append("  " + "; ".join(f"{u} = 0" for u in problem.inputs) + ";")
    for item in problem.timed_inputs:
        lines.append(f"  {item.target} := {item.formula};")
    readouts = [(name, emitted[name]) for name in species if emitted[name] != name]
    if readouts:
        lines.append("  // measurement mappings: the observables reference readout_*")
        lines += [f"  {name} := {sid};" for name, sid in readouts]
    for name, expression in zip(names, rendered):
        text = str(expression.subs({sp.Symbol(n): sp.Symbol(s) for n, s in readouts}))
        lines.append(f"  {emitted[name]}' = {text.replace('**', '^')};")
    if "clock" in species:
        lines.append("  clock' = 1;")
    lines += [f"  {symbol} = {value:.10g};" for symbol, value in constants]
    return "\n".join((*lines, "end", "")), len(constants)


def _data_units(staged: S.Staged, names, expressions) -> tuple[list[sp.Expr], list[tuple[str, float]]]:
    """Each normalised derivative rewritten in data units, with its non-integer constants named
    ``c0, c1, ...`` across the whole system, one name per magnitude."""
    substitutions = {sp.Symbol(v): sp.Symbol(v) / sp.Float(staged.scales[v]) for v in staged.field_inputs
                     if staged.scales[v] != 1.0}
    constants: list[tuple[str, float]] = []
    rendered = []
    for name, expression in zip(names, expressions):
        factor = staged.scales[name] / staged.horizon
        # Skip a unit factor of exactly 1, as the input substitutions above already do: it changes
        # nothing, but as a Float it would be named `cN` and refit, buying a parameter and its
        # ln(n) of BIC for a multiplication by one.
        #
        # Only EXACTLY one. An integral factor is not an integer: this number is the data's RMS
        # magnitude over the horizon, so a 2.0 is 2.0 by coincidence of the measurements, and
        # rounding it into the expression would present a fitted magnitude as model structure.
        scaled = expression.subs(substitutions)
        if factor != 1.0:
            scaled = sp.Float(factor) * scaled
        named, constants = _name_constants(sp.expand(scaled), constants)
        rendered.append(named)
    used = set().union(*({str(x) for x in e.free_symbols} for e in rendered)) if rendered else set()
    return rendered, [(s, v) for s, v in constants if s in used]


def _name_constants(expression: sp.Expr, existing: list[tuple[str, float]]) -> tuple[sp.Expr, list]:
    by_magnitude = {abs(v): s for s, v in existing}
    substitutions = {}
    for atom in sorted(expression.atoms(sp.Number), key=str):
        if atom.is_Integer:
            continue
        value = float(atom)
        magnitude, sign = abs(value), (-1 if value < 0 else 1)
        if magnitude not in by_magnitude:
            symbol = f"c{len(existing)}"
            existing = [*existing, (symbol, magnitude)]
            by_magnitude[magnitude] = symbol
        substitutions[atom] = sign * sp.Symbol(by_magnitude[magnitude])
    return expression.subs(substitutions), existing


def _species_id(name: str, states, problem: Problem) -> str:
    """The id a state is emitted under: its own, unless it carries a ``readout_`` prefix, which
    belongs to the assignment rule instead. Suffixed until it collides with nothing else named in
    the model -- another state, a condition column, a protocol target."""
    if not name.startswith(READOUT_PREFIX):
        return name
    taken = {*states, *problem.inputs, *(item.target for item in problem.timed_inputs)}
    candidate = base = f"s_{name[len(READOUT_PREFIX):]}"
    while candidate in taken:
        candidate = f"{base}_"
    return candidate


def _identifier(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"m_{cleaned}"
