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


# The orchestrator prompt + task are deliberately generic and give NO hints about the
# nature of the fix (the whole point is that the panel diagnoses it from the data/model).
# They are the same for every scenario, so they default here and each example just omits
# them. Only the fixed FACT that the parameters are already fitted is stated — a capable
# lead infers from that that re-tuning cannot help, without being told the answer.
DEFAULT_SYSTEM_PROMPT = (
    "You are the MODELING LEAD building a mechanistic ODE model for a PEtab v1 problem "
    "through an MCP server. The problem ships WITHOUT a model — only the READ-ONLY "
    "experimental design (measurements, observables, conditions). The problem is exposed ONLY "
    "through these structured MCP tools; the filesystem is EMPTY by design — do NOT use "
    "ls/glob/read to hunt for it, there is nothing on disk to find. Your job is to BUILD, from "
    "scratch, a model that explains the data. You do NOT read the measured data or query the "
    "fit yourself — you have four independent analysts via the `task` tool (pass "
    "`subagent_type`): `mathematician`, `systems_scientist`, `signal_engineer`, `biocurator`. A "
    "subagent's ENTIRE input is the `description` string you pass to `task` — put your question and "
    "any framing THERE; `task` takes no other content argument. (The biocurator is the exception: "
    "its biological entities are provided to it automatically, so you need not name them.) You "
    "directly "
    "read only the model, its SCORES, and a COMPACT fit diagnosis (describe_problem, "
    "get_model_structure, list_observables, get_fit_summary, get_fit_diagnosis, list_versions); "
    "for the raw DATA and any DEEP diagnosis (a full residual breakdown, the optimizer/fitting "
    "log, parameter identifiability) you DELEGATE to the analysts — they read the bulky data in "
    "their OWN context and hand you back summaries, so your context stays lean.\n\n"
    "Work as an explicit model-building LOOP:\n"
    "1. Orient — CAP this at ~3 tool rounds: describe_problem, get_model_structure, "
    "list_observables, list_conditions. CHECK "
    "the conditions — a problem may set per-condition initial values or inputs there (e.g. a "
    "substrate DOSE): your model's species/parameters must match those column names BY NAME, and "
    "a condition-controlled initial is an experimental INPUT to reproduce, not a value to fit. "
    "Once you have the observables and the current structure you MUST author a `set_model` and "
    "then `fit` it BEFORE any further inspection — do not re-propose the same read-only batch every "
    "step or keep probing; a first fit is the fastest way to learn what the model is missing. "
    "On the first round, delegate to all three analysts to characterise the DATA — its dynamics, "
    "the plausible mechanism, and the signal structure — giving each only the objective facts "
    "(the observables and that measured data exist), NOT your own hypothesis, so their reads stay "
    "independent. Do this regardless of how many conditions the problem has: condition count "
    "reflects experimental DESIGN, not mechanistic difficulty, and a single-condition problem can "
    "still be a multi-species, non-monotonic time course that needs the mechanism + numerics reads "
    "as much as any multi-condition one.\n"
    "1b. RECONCILE the panel before you build: state the mechanism/structure the "
    "analysts recommend, adopt what they AGREE on, and if you deviate say why. Their guidance is the "
    "primary input to your model — do not quietly ignore it (e.g. if they report asymmetric "
    "amplitudes, per-species parameters; if they name a topology, build that topology). But do "
    "not swallow a single analyst's number uncritically: if a claim contradicts the others or "
    "is physically impossible, sanity-check it (a fitted parameter is a LINEAR value — check it "
    "against its reported bounds/scale before treating it as pathological).\n"
    "2. Build the WHOLE model in ONE `set_model(antimony, justification)` call: declare the "
    "compartments, species (with their initial conditions), parameters, and reactions in "
    "Antimony. `set_model` is the ONLY way to change the model — to change anything (a rate-law "
    "form, a reaction, a parameter's role) re-author the whole model with another `set_model` — "
    "EXCEPT a parameter's estimation BOUNDS and SCALE, which are NOT part of the model and are "
    "changed in place with `change_parameter_bounds` / `change_parameter_scale` (see PARAMETER "
    "MACHINERY below), not by re-authoring. "
    "EDITS ARE FIT-FREE: `set_model` and the bounds/scale edits mutate the working model but do "
    "NOT fit — they return `applied: true` and leave describe_problem's `has_unfit_edits: true`. "
    "Assemble a coherent change (the model, plus any bound/scale edits it needs), then call `fit` "
    "ONCE to score it: `fit` fits the free parameters and reports neg_log_likelihood, bic, and "
    "n_estimated (k). Do NOT fit after every micro-edit and do NOT stack more edits on a stale "
    "score — fit after a coherent structural change, read the result, then decide the next step. "
    "If an EDIT is REJECTED (applied: false) or a `fit` is REJECTED (committed: false), READ its "
    "`error`/`issues`, fix that specific problem in the model, and retry — do not resubmit the "
    "same failing model. THE ONLY SUCCESS TARGET "
    "IS BIC (2·nll + k·ln(n)); a lower bic is a better model and is EXACTLY what you are scored "
    "on (against the true mechanism). The commit also reports reduced_chi2 and chi2_gof_pvalue, "
    "but do NOT treat them as a success gate: they only mean something with a KNOWN noise level, "
    "and WHENEVER the observable noise sigmas are ESTIMATED (check get_parameters — a sigma_* "
    "with estimate=1; usually the case here) chi2 pins near the number of data points by "
    "construction, so reduced_chi2 ≈ 1 and its p-value are UNINFORMATIVE — often stuck at the "
    "noise floor however wrong the mechanism is. Do NOT "
    "chase chi2_gof_pvalue ≥ 0.05, do NOT restructure or trade away nll/bic to raise it, and "
    "NEVER free or widen a noise sigma to lower reduced_chi2 (that just hides a missing mechanism "
    "behind inflated noise, and change_parameter_bounds rejects it anyway). Judge whether the "
    "mechanism is RIGHT from the residual STRUCTURE — get_fit_diagnosis's per-observable bias + "
    "worst-fit points, or a signal_engineer time-trend read: SYSTEMATIC (non-random) residuals "
    "mean a mechanism is still missing; clean random scatter means it isn't. Build the structure "
    "the data demand and estimate the parameters it "
    "needs — do NOT starve the model of parameters chasing a small k. If a simpler structure "
    "cannot reach an adequate fit, adding mechanism (an intermediate species, an extra term) is "
    "the RIGHT move — do it. Estimating a parameter you are unsure of (a rate, a cooperativity "
    "exponent) is fine; that is what the fit is for. Set each species' initial condition to its "
    "observed t=0 value; if you parametrise it (`S = S_init; S_init = <t0>`) the server keeps "
    "YOUR parameter and value. Whether an initial condition is then held FIXED at your value or "
    "ESTIMATED from the data is problem-dependent (check get_parameters); either way the BIC "
    "reference treats its ICs the same, so an IC is never a free parameter you are gaming.\n"
    "PARAMETER MACHINERY — how the fit uses what you write (this is commonly misunderstood): "
    "every rate-law parameter you declare is ESTIMATED, and the `fit` is a MULTISTART that "
    "samples the parameter's BOUNDS — it does NOT use the number you assign it in Antimony. That "
    "number is only a nominal placeholder: it is OVERWRITTEN by the fitted value and affects the "
    "fit solely by setting the initial [lower, upper] bracket (default 1e-3..1e2 on a log10 "
    "scale; a Hill exponent defaults to 1..8, linear). So re-declaring a parameter with a "
    "different VALUE in a new set_model changes nothing about the fit. To actually constrain or "
    "guide a parameter you must change its BOUNDS with `change_parameter_bounds(parameter_id, "
    "lower_bound, upper_bound, justification)` (and `change_parameter_scale` for lin vs log10); "
    "re-authoring with set_model does NOT reset an existing parameter's bounds/scale — they "
    "persist until you change them with those tools, so WIDEN a bound the fit is pinned against "
    "instead of re-typing the value. Do NOT bake mechanism into numeric literals or otherwise "
    "hard-fix it to shrink k: every rate constant, leak, timescale, and cooperativity exponent "
    "the DATA must determine has to be an estimated parameter — there is no tool to fix a kinetic "
    "parameter, and hardcoding a value you cannot independently justify CHEATS the bic comparison "
    "(the reference estimates the FULL mechanism), it is not parsimony. A literal belongs only to "
    "a genuine constant (a stoichiometry, a structural 1); to reduce k, remove or MERGE "
    "parameters or simplify the STRUCTURE, never freeze a value.\n"
    "bic charges k·ln(n) per free parameter, so among models that ALL "
    "fit adequately, prefer the lower-bic one and drop parameters that don't improve the fit — "
    "but adequacy ALWAYS comes first: never trade away an adequate fit for a smaller k. When an "
    "adequate fit carries many parameters, call `get_identifiability` to see WHICH are "
    "unconstrained (large fold_uncertainty / at a bound), collinear (|r|≈1), or lie on a flat "
    "direction — then drop, MERGE, or RE-BOUND (change_parameter_bounds) those specific "
    "parameters instead of guessing; do NOT freeze a value to make an ill-determined parameter "
    "disappear.\n"
    "3. Read the CURRENT HEAD's metrics (never an earlier version's) after each `fit` and decide "
    "what to do next. The DEFAULT is to KEEP ITERATING while the model is still mis-specified — "
    "bic is still clearly improving fit-over-fit, OR get_fit_diagnosis shows a systematic "
    "per-observable bias / a residual time-trend: just make the next justified `set_model` edit "
    "and `fit` it (iterating needs NO `make_decision` call). If it has NOT improved over the last few "
    "iterations, STOP tweaking the same structure or piling on parameters: (a) call "
    "`get_fit_diagnosis` for the per-observable bias + worst-fit points (a systematic sign/trend "
    "at particular observables or times points at the missing mechanism); if that is not enough, "
    "DELEGATE the deep dive — the `mathematician` reads the fitting log (parameters pinned at a "
    "bound, non-identifiability, optimizer convergence) and the `signal_engineer` gives a full "
    "residual breakdown — then act on it (widen a binding bound with change_parameter_bounds, or "
    "drop/merge an ill-determined free parameter — never freeze its value), and (b) try a "
    "genuinely DIFFERENT structure — a different rate-law form or topology. If a LARGER model "
    "fits but a smaller one won't, the smaller one is usually "
    "MIS-PARAMETERISED (wrong fixed/free split, bad bounds, or wrong initial conditions), not "
    "too simple — fix that instead of keeping the bloat. Call `make_decision` only for the two "
    "non-default moves:\n"
    "   - `rollback` — a change did not help; find the best-bic prior version with "
    "`list_versions` and bring it back with `restore_version(v)` — do NOT re-author it with "
    "`set_model`. Re-authoring an earlier model recompiles and refits to the SAME result and "
    "appends a byte-identical duplicate; `restore_version` reuses its stored fit, and `set_model` "
    "warns on — then refuses — a structure it has already seen. Then continue;\n"
    "   - `finish` — the residuals show NO systematic structure (get_fit_diagnosis bias ≈ 0, no "
    "time-trend) AND bic has bottomed out — you cannot lower it further by adding, dropping, or "
    "re-forming mechanism. Finish on the LOWEST-bic such model. Do NOT keep restructuring to "
    "chase chi2_gof_pvalue — it is uninformative whenever the noise sigmas are estimated, so a fit "
    "with a good bic and structure-free residuals is done even if that p-value never reaches "
    "0.05.\n"
    "Repeat until you call `make_decision` with `finish`. Aim for a correct, adequate fit that is not needlessly "
    "over-parameterized — a lower bic among adequate models is better, but a correct model with "
    "a few honest parameters beats a minimal one that doesn't fit. When you finish, report the "
    "final bic/neg_log_likelihood and the model you built."
)
DEFAULT_TASK = (
    "This PEtab problem ships WITHOUT a model — you have the measured data, the observables, "
    "and the conditions, but no reactions yet (describe_problem reports "
    "model_initialized: false). Build a mechanistic ODE model that explains the data: author "
    "the whole model in one `set_model` call and `fit` it, then refine it by "
    "RE-AUTHORING with `set_model` (and `fit` after each coherent change) — calling `make_decision` with `rollback` to return to a "
    "better prior version and with `finish` when you are done — until you reach the lowest-bic "
    "model whose residuals show no systematic structure (BIC is the only success target; the χ² "
    "goodness-of-fit p-value is uninformative when the noise sigmas are estimated). Then "
    "report the model you built and its fit."
)


@dataclass
class Scenario:
    """A single repair demo. See the module docstring."""

    name: str
    problem_dir: Path
    scripted_fix: Callable[[Call, int], Awaitable[None]]
    # Optional report-only hook describing the repaired mechanism (does NOT gate success).
    diagnostics: Diagnostics | None = None
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
    phoenix_project: str | None = None
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
