"""The :class:`Scenario` contract + the judge toolkit shared by every example.

A *scenario* is everything that distinguishes one repair demo from another: the pristine
(broken) PEtab problem, the orchestrator prompt + task string handed to the modeling lead,
a deterministic ``scripted_fix`` (the canonical repair, used by ``--driver scripted`` as a
no-LLM self-test), a ``bic_threshold``, and an optional ``diagnostics`` hook. Success itself is
decided generically by the runner's BIC gate (``delta_bic <= bic_threshold`` in ``_judge``), not
by the scenario; ``diagnostics`` is report-only. Everything else — cleanup, the stateful MCP
server, the deepagents panel, Phoenix tracing, baseline scoring, the RESULT report, and
``result.json`` — lives in :mod:`runner` and is identical across examples.

The judge helpers below let a scenario's ``diagnostics`` hook reason about the *structure* of a fitted rate
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

# The lead's prompt + task moved to prompts.py, where the WHOLE agent-facing prompt surface now
# lives. Re-exported here because they are this dataclass's field defaults and callers (and the
# Phoenix prompt registry) have always read them off this module.
from .prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_TASK

# A bound MCP caller: ``await call("get_fit_summary")`` / ``await call("set_model",
# antimony=..., justification=..., references=[...])``.
Call = Callable[..., Awaitable[Any]]


# --------------------------------------------------------------------------- data classes
@dataclass
class Baseline:
    """The initial (pre-repair) state the judge compares against.

    ``nll`` is ``None`` for a zero-shot problem that ships without a model (there is no
    baseline fit to score until the agent builds one)."""

    nll: float | None
    species: set[str]
    structure: dict[str, Any]


@dataclass
class JudgeContext:
    """Everything a scenario's ``diagnostics`` hook needs to describe the repaired model.

    ``call`` is the bound MCP caller; ``baseline`` is the pre-repair snapshot; the rest is
    the current (post-repair) state the runner has already gathered (from get_fit_summary +
    get_model_structure) so a hook doesn't re-fetch it. Success itself is decided
    generically by the runner's BIC gate, not here — diagnostics is report-only.
    """

    call: Call
    baseline: Baseline
    final_nll: float
    structure: dict[str, Any]
    new_species: list[str]
    delta: float
    n: int            # number of data points
    k: int            # number of estimated (fitted) parameters
    chi2: float | None


# A diagnostics hook: given the post-repair state, return (report_lines, extra_fields)
# describing the MECHANISM the agent landed on (e.g. the rate-law form). Report-only — it
# does NOT decide success; the BIC gate does. ``None`` skips it.
Diagnostics = Callable[[JudgeContext], Awaitable["tuple[list[str], dict[str, Any]]"]]


def bic(nll: float, k: int, n: int) -> float:
    """Bayesian Information Criterion from a neg-log-likelihood: 2·nll + k·ln(n)."""
    import math

    return 2.0 * float(nll) + int(k) * math.log(max(int(n), 1))


def chi2_gof_pvalue(chi2: float | None, dof: int) -> float | None:
    """Upper-tail p-value of a χ² goodness-of-fit test, or ``None`` if not computable.

    ``chi2`` is the sum of squared noise-weighted residuals (what the server reports). Under
    a correctly-specified model with the given (known) noise it is χ²-distributed with
    ``dof = n - k`` degrees of freedom, so a small p-value means the model fits significantly
    worse than the noise allows (rejectable on goodness-of-fit grounds).
    """
    if chi2 is None or dof <= 0:
        return None
    try:
        from scipy.stats import chi2 as _chi2

        return float(_chi2.sf(float(chi2), dof))
    except Exception:
        return None


async def build_true_model(
    call: Call,
    *,
    antimony: str,
    justification: str,
    references: list[str] | None = None,
    n_starts: int,
    changes: list[dict] | None = None,
    change_justification: str = "",
    min_n_starts: int | None = None,
) -> None:
    """The canonical ``scripted_fix`` body: set_model the true structure, optionally re-bound
    some parameters, then fit once — reporting whatever the server says at each step.

    Every example's ``scripted_fix`` was a copy of this (16 of them, 10 byte-identical). What
    actually varies is data, so it is passed in: the ``antimony`` to build, its ``justification``
    / ``references``, and two optional knobs a handful of examples need.

    ``changes`` is a ``change_parameters`` list applied between the build and the fit, for a
    model whose kinetics need a tighter box than the server's wide default before the multistart
    can find the true basin. The caller builds the list, because how it is derived is
    scenario-specific (a flat bound pair, per-parameter bounds, a merged scale+bound entry).

    ``min_n_starts`` floors the multistart count, for a sharp likelihood the default 50 starts
    cannot reliably close. It only ever raises ``n_starts``, never lowers a caller's request.
    """
    if min_n_starts is not None:
        n_starts = max(int(n_starts), min_n_starts)

    res = await call("set_model", antimony=antimony, justification=justification,
                     references=references)
    if isinstance(res, dict) and res.get("applied") is False:
        print(f"[scripted] set_model rejected: {res.get('error')} / {res.get('issues')}")
        return

    if changes:
        await call("change_parameters", parameters=changes, justification=change_justification)

    res = await call("fit", n_starts=n_starts)
    if isinstance(res, dict) and res.get("committed") is False:
        print(f"[scripted] fit rejected: {res.get('error')} / {res.get('issues')}")
    elif isinstance(res, dict):
        print(f"[scripted] built model v{res.get('version')}; nll={res.get('neg_log_likelihood')} "
              f"fitted={res.get('fitted_parameters')}")
    else:
        print(f"[scripted] fit returned: {res}")


async def topology_diagnostics(ctx: "JudgeContext") -> tuple[list[str], dict]:
    """Report-only: how many species and reactions the agent built (success is decided by BIC).

    The default :attr:`Scenario.diagnostics`. Eight examples each defined their own copy of
    exactly this, differing only in whether the species names were interpolated into the line;
    they are, here, since the judge already records the full lists either way
    (``runner.judge`` writes ``species``/``reactions``/``new_species`` into ``result.json``).
    An example with something real to say about its mechanism overrides this — see the
    rate-law analysis in the synthetic examples' own hooks.
    """
    reactions = ctx.structure.get("reactions", []) or []
    species = ctx.structure.get("species", []) or []
    lines = [
        f"species built:        {len(species)} ({', '.join(map(str, species))})",
        f"reactions built:      {len(reactions)}",
    ]
    return lines, {"n_species_built": len(species), "n_reactions_built": len(reactions)}


@dataclass
class Scenario:
    """A single repair demo. See the module docstring."""

    name: str
    problem_dir: Path
    scripted_fix: Callable[[Call, int], Awaitable[None]]
    # Report-only hook describing the repaired mechanism (does NOT gate success). Defaults to the
    # species/reaction count every benchmark example used to spell out for itself; override it
    # when the example can say something specific about its own mechanism, or set it to None to
    # report nothing.
    diagnostics: Diagnostics | None = topology_diagnostics
    # The orchestrator prompt + task default to the generic, hint-free versions above;
    # scenarios only override them if they truly need to (they don't).
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    task: str = DEFAULT_TASK
    # A scenario-specific note appended to the task. For benchmark problems that SHIP known,
    # fixed constants (a measured baseline, a burst size, an observable scaling), it explains
    # what each is and asks the agent to include it BY NAME (fixed, not estimated) — the same
    # information a modeller would be handed. Empty for the synthetic examples, which hide it.
    known_constants: str = ""
    # Hand-curated biological entity names handed to the blinded `biocurator` subagent (issue
    # #91). The biocurator holds no petab tool and cannot fetch its own input like the analysts,
    # so the harness injects these DIRECTLY into its prompt. Curate clean, database-lookupable
    # names (genes/proteins/metabolites/cell types), NOT the technical observable IDs (`obs_P`).
    #
    # STRICT BLINDING RULE — include ONLY what a modeller is already handed, never the mechanism to
    # rediscover: (1) entities that are MEASURED (appear in an observable), and (2) experimental
    # inputs SET in the condition table (a dose, a stimulus, a drug, a genotype switch). Do NOT list
    # a LATENT/unobserved model species — an unmeasured intermediate, complex, or driver (e.g.
    # crauste's Pathogen, an EGF_EGFR complex): that species IS the hidden mechanism, so naming it
    # both leaks the answer AND pre-empts the biocurator's core job (surfacing those hidden
    # entities from the measured ones). Leave EMPTY for abstract/synthetic problems with no
    # lookupable biology (predator–prey, a toy repressilator, generic enzyme kinetics): the
    # biocurator is then handed no entities (nothing to look up).
    biocurator_entities: list[str] = field(default_factory=list)
    # Success gate: a repair is accepted iff it can't be rejected against the true model —
    # BIC(candidate) - BIC(true) <= bic_threshold. ~10 = "very strong evidence against"
    # (Kass & Raftery); the verdict is insensitive here (genuine repairs sit at ΔBIC≈0,
    # broken/near-miss models are hundreds off).
    bic_threshold: float = 10.0
    # Short human description of the canonical repair (for the scripted driver's log).
    fix_summary: str = ""
    # Example definition version. BUMP THIS whenever you change what the example *is* — its
    # planted bug, data (generate_data.py / problem/), prompts, canonical fix, or judge — so
    # runs of different definitions are not silently pooled. It is independent of the repo
    # git sha (the harness/code version): the git sha moves on ANY commit, while `version`
    # moves only when THIS example changes, so it groups comparable runs across harness
    # commits. Recorded on every run (Phoenix span + result.json + metrics history).
    version: int = 1

    @property
    def example_dir(self) -> Path:
        """The example root (holds ``problem/``; ``work/`` + ``.amici_build/`` go here)."""
        return self.problem_dir.parent

    @property
    def agent_task(self) -> str:
        """The task handed to the deepagents driver: the base task plus any scenario-specific
        ``known_constants`` note (so the agent is told about shipped fixed constants)."""
        return f"{self.task}\n\n{self.known_constants}" if self.known_constants else self.task

    def reference(self) -> dict[str, Any] | None:
        """The true-model fit reference {nll, k, n} from ``problem/reference.json``.

        Written by the example's ``generate_data.py`` (it fits the TRUE structure to the
        data). ``None`` if absent — the runner then can't compute ΔBIC and says so.
        """
        import json

        path = self.problem_dir / "reference.json"
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None


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
