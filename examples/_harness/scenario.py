"""The :class:`Scenario` contract + the judge toolkit shared by every example.

A *scenario* is everything that distinguishes one repair demo from another: the pristine
(broken) PEtab problem, the orchestrator prompt + task string handed to the modeling lead,
a deterministic ``scripted_fix`` (the canonical repair, used by ``--driver scripted`` as a
no-LLM self-test), and an independent ``check`` that decides success. Everything else —
cleanup, the stateful MCP server, the deepagents panel, Phoenix tracing, baseline scoring,
the RESULT report, and ``result.json`` — lives in :mod:`runner` and is identical across
examples.

The judge helpers below let each ``check`` reason about the *structure* of a fitted rate
law rather than string-matching it, so a repair is credited for the right MECHANISM
however the agent chose to name or arrange its new parameters. They parse the libsbml
formula with sympy and ask questions like "is this reaction's rate saturating in species
X?" (X appears in the denominator) or "is this repression cooperative?" (the repressor's
effective exponent in the denominator is ≥ 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# A bound MCP caller: ``await call("get_fit_summary")`` / ``await call("change_rate_law",
# reaction_id=..., rate_law=..., justification=..., references=[...])``.
Call = Callable[..., Awaitable[Any]]


# --------------------------------------------------------------------------- data classes
@dataclass
class Baseline:
    """The initial (pre-repair) state the judge compares against."""

    nll: float
    species: set[str]
    structure: dict[str, Any]


@dataclass
class JudgeContext:
    """Everything a scenario's ``check`` needs to decide success.

    ``call`` is the bound MCP caller; ``baseline`` is the pre-repair snapshot; the rest is
    the current (post-repair) state the runner has already gathered so every ``check``
    doesn't re-fetch it.
    """

    call: Call
    baseline: Baseline
    final_nll: float
    structure: dict[str, Any]
    new_species: list[str]
    delta: float


@dataclass
class Verdict:
    """A scenario's success decision, plus material for the report + result.json."""

    success: bool
    verdict: str
    fields: dict[str, Any] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    """A single repair demo. See the module docstring."""

    name: str
    problem_dir: Path
    system_prompt: str
    task: str
    scripted_fix: Callable[[Call, int], Awaitable[None]]
    check: Callable[[JudgeContext], Awaitable[Verdict]]
    # A meaningful fit gain: the baseline ships pre-fitted, so the objective only drops
    # markedly once the STRUCTURE is right. Tuned per scenario to the known broken↔true gap.
    min_delta: float = 1.0
    # Short human description of the canonical repair (for the scripted driver's log).
    fix_summary: str = ""
    phoenix_project: str | None = None

    @property
    def example_dir(self) -> Path:
        """The example root (holds ``problem/``; ``work/`` + ``.amici_build/`` go here)."""
        return self.problem_dir.parent


# --------------------------------------------------------------------------- rate-law tools
def reaction_rate_law(structure: dict[str, Any], reaction_id: str) -> str | None:
    """The current rate-law string of ``reaction_id`` from ``get_model_structure``."""
    for r in structure.get("reactions", []) or []:
        if r.get("id") == reaction_id:
            return r.get("rate_law")
    return None


def fitted_values(param_records: list[dict[str, Any]]) -> dict[str, float]:
    """Map ``parameterId -> nominalValue`` (the fitted values) from ``get_parameters``."""
    from petab.v1.C import NOMINAL_VALUE, PARAMETER_ID

    out: dict[str, float] = {}
    for rec in param_records or []:
        pid = rec.get(PARAMETER_ID) or rec.get("parameterId") or rec.get("id")
        val = rec.get(NOMINAL_VALUE, rec.get("nominalValue"))
        if pid is None or val is None:
            continue
        try:
            out[str(pid)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _parse(rate_law: str | None):
    """Parse a libsbml infix rate law into a sympy expression (``None`` on failure).

    libsbml L3 formulae use ``^`` for exponentiation and ``pow(a, b)`` for power calls;
    sympy wants ``**``. Every non-call identifier is bound to a plain ``Symbol`` before
    sympifying so that reserved single letters — most importantly a species named ``S``
    (which otherwise resolves to sympy's ``S`` singleton registry, not a symbol), but also
    ``E``, ``I``, ``N``, ``O``, ``Q`` — are treated as ordinary model symbols.
    """
    if not rate_law:
        return None
    import re

    import sympy

    text = str(rate_law).replace("^", "**")
    locals_: dict[str, Any] = {}
    for m in re.finditer(r"[A-Za-z_]\w*", text):
        name = m.group(0)
        # Leave function calls (name immediately followed by '(') to sympy, so pow(...),
        # exp(...) etc. keep their meaning; bind everything else as a bare Symbol.
        if text[m.end():].lstrip().startswith("("):
            continue
        locals_[name] = sympy.Symbol(name)
    try:
        expr = sympy.sympify(text, locals=locals_, evaluate=True)
    except Exception:
        return None
    # Rewrite any pow(a, b) function calls (some libsbml versions emit these) to a**b.
    pow_fn = sympy.Function("pow")
    expr = expr.replace(lambda n: getattr(n, "func", None) == pow_fn,
                        lambda n: sympy.Pow(*n.args))
    return expr


def _numerator(expr):
    """The numerator of ``expr`` after putting it over a common fraction."""
    import sympy

    return sympy.fraction(sympy.together(expr))[0]


def _denominator(expr):
    """The denominator of ``expr`` after putting it over a common fraction."""
    import sympy

    return sympy.fraction(sympy.together(expr))[1]


def _max_power_of(part, symbol: str, fitted: dict[str, float] | None) -> float:
    """Effective maximum exponent of ``symbol`` in a (numerator or denominator) expression.

    An explicit power whose BASE involves the symbol — ``S^2``, ``p3^n``, ``(p3/K3)^n`` —
    carries the exponent (matching "sym in base", not "base == sym", catches the scaled Hill
    form); a symbolic exponent is resolved from ``fitted``. With no explicit power, falls
    back to the polynomial degree (catches expanded forms like ``Km + S + S*S/Ki`` → 2).
    Returns 0 if the symbol is absent.
    """
    import sympy

    fitted = fitted or {}
    sym = sympy.Symbol(symbol)
    if sym not in part.free_symbols:
        return 0.0
    subs = {sympy.Symbol(k): v for k, v in fitted.items()}
    powers: list[float] = []
    for node in sympy.preorder_traversal(part):
        if isinstance(node, sympy.Pow) and sym in node.base.free_symbols:
            exponent = node.exp
            if exponent.is_number:
                powers.append(float(exponent))
            else:
                try:
                    powers.append(float(exponent.subs(subs)))
                except (TypeError, ValueError):
                    powers.append(1.0)
    if powers:
        return max(powers)
    try:
        return float(sympy.degree(sympy.expand(part), gen=sym))
    except Exception:
        return 1.0  # present but bare (power 1)


def is_rational(rate_law: str | None) -> bool:
    """True if the rate law is a genuine rational function (has a non-constant denominator).

    A saturating law (Michaelis–Menten, Holling type II, Hill) divides by a state-dependent
    denominator; mass action / bilinear predation does not.
    """
    expr = _parse(rate_law)
    if expr is None:
        return "/" in str(rate_law or "")
    try:
        denom = _denominator(expr)
        return len(denom.free_symbols) > 0
    except Exception:
        return "/" in str(rate_law)


def denominator_has_symbol(rate_law: str | None, symbol: str) -> bool:
    """True if ``symbol`` (a species) appears in the rate law's denominator.

    Distinguishes a rate that SATURATES in that species (X in the denominator, e.g. a
    Holling type-II response a*X*Y/(1+a*h*X)) from a non-saturating one (bilinear a*X*Y).
    """
    import sympy

    expr = _parse(rate_law)
    if expr is None:
        return False
    try:
        denom = _denominator(expr)
        return sympy.Symbol(symbol) in denom.free_symbols
    except Exception:
        return False


def max_power_in_denominator(
    rate_law: str | None, symbol: str, fitted: dict[str, float] | None = None
) -> float:
    """Effective maximum exponent of ``symbol`` in the rate law's denominator.

    Michaelis–Menten / Holling II are linear in the saturating species (power 1). A Hill /
    cooperative law carries it to a power > 1 — either a literal ``p^2`` (power 2) or a
    symbolic ``p^n`` whose fitted exponent ``n`` we resolve from ``fitted``. Substrate
    inhibition likewise pushes the denominator to degree ≥ 2 in the substrate (``... +
    S^2/Ki``). Returns 0 if the symbol is absent from the denominator.
    """
    expr = _parse(rate_law)
    if expr is None:
        return 0.0
    try:
        return _max_power_of(_denominator(expr), symbol, fitted)
    except Exception:
        return 0.0


def max_power_in_numerator(
    rate_law: str | None, symbol: str, fitted: dict[str, float] | None = None
) -> float:
    """Effective maximum exponent of ``symbol`` in the rate law's numerator.

    Paired with :func:`max_power_in_denominator`, this separates a rate that *saturates* or
    *falls* at high ``symbol`` from one that keeps growing. A rate DECAYS at large ``symbol``
    (the substrate-inhibition signature — the rate rises then falls) exactly when the
    denominator outgrows the numerator in that symbol; a Michaelis–Menten (1/1) or Hill
    ``S^n/(K^n+S^n)`` (n/n) law has equal degrees and only saturates (monotone). Returns 0
    if the symbol is absent from the numerator.
    """
    expr = _parse(rate_law)
    if expr is None:
        return 0.0
    try:
        return _max_power_of(_numerator(expr), symbol, fitted)
    except Exception:
        return 0.0


async def get_fitted_values(call: Call) -> dict[str, float]:
    """Fetch the current fitted parameter values via ``get_parameters``."""
    return fitted_values(await call("get_parameters"))
