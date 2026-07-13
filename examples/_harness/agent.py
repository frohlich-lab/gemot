"""deepagents driver: a panel of three independent analyst subagents + a modeling lead.

This module is **scenario-agnostic** — the same panel drives every example in
``examples/`` (enzyme kinetics, predator–prey, the repressilator, …). The model-analysis
phase is split across three analysts that each look at the problem through a different
lens and report to a single **orchestrator** ("modeling lead"). deepagents exposes the
analysts to the orchestrator as a ``task`` tool: the lead calls
``task(description=..., subagent_type="mathematician" | "systems_scientist" |
"signal_engineer")``, each analyst runs its own reasoning loop over *its own curated tool
subset*, and returns a written report. The lead synthesizes the three reports and is the
only agent that edits the model.

Why three agents instead of one:

* **mathematician** — equation structure, dynamical-systems / chemical-reaction-network
  theory, and the numerics of parameter estimation. It reads the model equations, the
  parameter search space, and the pyPESTO/Fides *fitting log* to judge whether a misfit is
  structural (a rate-law FORM that cannot produce the observed dynamics, a broken
  conservation law) or numerical (non-identifiability, a start pinned at a bound).
* **systems_scientist** — mechanistic plausibility. It reads the mechanism, the measured
  data, and the parameter magnitudes and judges whether a proposed mechanism is how the
  system actually works (enzyme saturation, cooperative regulation, a saturating predator
  functional response, degradation, feedback) rather than a curve-fitting artifact.
* **signal_engineer** — the residuals and trajectories as *signals*. It reads the
  measured-vs-simulated points and the residual breakdown and reports *where* and *how* the
  fit fails (bias, drift, curvature/timing mismatch, a missed oscillation) without claiming
  a mechanism.

The personas are written to teach the analyst HOW to reason about a mis-specified dynamic
model — they carry a balanced set of archetypes drawn from enzyme kinetics, gene
regulation, and ecology so the panel generalizes across examples without being handed any
single problem's answer. The scenario-specific framing lives entirely in the orchestrator
prompt + task string that ``runner.run_scenario`` passes in.

Each analyst is deliberately given only the tools its lens needs (see the ``*_TOOLS`` sets
below), so its context is not bloated with the full tool surface and its report stays in
character. The union of all analyst tools plus the lead's own editing/recovery tools is
handed to ``create_deep_agent`` (deepagents resolves each subagent's tool subset by name
from that union).

Observability is handled by **Arize Phoenix tracing** (see ``tracing.py``), which the
runner turns on before the agent starts.
"""

from __future__ import annotations

import inspect
from typing import Any

# ---------------------------------------------------------------- tool exposure
# Each analyst persona sees ONLY the tools its lens needs. Curating per-persona (rather
# than handing every agent the full tool surface) keeps each context small and keeps the
# analyst in character: the signal engineer reasons about residuals, not equations; the
# systems scientist about mechanism, not optimizer internals; and so on.

# Mathematician: equation structure, dynamical-systems theory, numerical stability. It
# needs the model equations, the parameter search space, and — uniquely — the pyPESTO/Fides
# fitting log to tell a structural misfit apart from a numerical one.
MATHEMATICIAN_TOOLS = [
    "get_model",            # the rate-law equations (Antimony / SBML)
    "get_model_structure",  # species, reactions, rate laws — the reaction network
    "get_parameters",       # scales, bounds — identifiability / conditioning
    "validate_problem",
    "get_fit_summary",
    "get_fitting_log",      # pyPESTO/Fides log: multistart spread, convergence, bounds
    "list_versions",
    "diff_versions",
]

# Systems scientist: is the mechanism real? Needs the mechanism as written, what was
# actually measured (the readout), and whether rate magnitudes are plausible — NOT the
# optimizer internals.
SYSTEMS_SCIENTIST_TOOLS = [
    "describe_problem",
    "get_model",            # the mechanism as written
    "get_model_structure",
    "list_observables",     # what quantity is measured
    "list_conditions",      # the experimental setup
    "get_measurements",     # the actual data / dynamic behaviour
    "get_parameters",       # are the rate magnitudes plausible?
]

# Signal-processing engineer: residuals and trajectories as signals. Needs measured-vs-
# simulated points and the residual breakdown; it localizes the misfit (bias / drift /
# curvature / a missed oscillation) but leaves the *mechanism* to the other two.
SIGNAL_ENGINEER_TOOLS = [
    "list_data_tables",     # the queryable tables (simulations, residuals, ...) + columns
    "query_data",           # SQL over measured-vs-simulated points + residual breakdowns
    "get_measurements",     # the raw signal
    "list_observables",     # the noise model
    "get_fit_summary",
]

# The orchestrator ("modeling lead") orients, delegates analysis to the three analysts, and
# is the ONLY agent that edits the model or recovers. It keeps a light read surface for
# orientation + scoring plus the editing/recovery tools; the deep diagnosis is the
# analysts' job. Batching (begin_batch/commit_batch/abort_batch) lets it land a
# multi-reaction fix as ONE fitted version — essential when a single mechanism spans
# several reactions (a ring of repressions, a functional response that appears in two
# fluxes).
ORCHESTRATOR_TOOLS = [
    # orient / score
    "describe_problem",
    "get_model",
    "get_model_structure",
    "get_fit_summary",
    "current_version",
    "list_versions",
    "diff_versions",
    "get_errors",
    # edit the model (lead only)
    "init_model",           # zero-shot a whole model when none exists (refused once one does)
    "add_reaction",
    "change_rate_law",
    "remove_reaction",
    "change_parameter_bounds",
    "change_parameter_scale",
    # batch several related edits into one fitted version
    "begin_batch",
    "commit_batch",
    "abort_batch",
    # recover / audit
    "rollback_to",
    "reset",
]

# The full set handed to create_deep_agent: the union of the lead's tools and every
# analyst's tools (deepagents resolves each subagent's `tools` names from this set).
ALLOWED_TOOLS = set(
    ORCHESTRATOR_TOOLS
    + MATHEMATICIAN_TOOLS
    + SYSTEMS_SCIENTIST_TOOLS
    + SIGNAL_ENGINEER_TOOLS
)


# ---------------------------------------------------------------- analyst personas
# System prompts for the three analysts. Each is a focused expert that INSPECTS (it has no
# editing tools) and returns a concise written report ending in a concrete recommendation
# the lead can act on. They are independent: none sees the others' reports.
_REPORT_CONTRACT = (
    "\n\nYou cannot edit the model — you only inspect and advise. Investigate with your "
    "tools, then return a SHORT written report: (1) what you found, (2) the single most "
    "likely explanation from your perspective, (3) a concrete recommendation for the "
    "modeling lead (name the operation and, if you can, the exact reaction / rate law). If "
    "the fix spans several reactions (e.g. a functional response that appears in two fluxes, "
    "or a symmetric ring), say so and name every reaction that must change together. Flag "
    "explicitly if the evidence is ambiguous or outside your expertise."
)

MATHEMATICIAN_PROMPT = (
    "You are a mathematician specializing in ODE models of reaction networks and population "
    "dynamics: rate-law structure, bifurcations, and the numerics of parameter estimation. "
    "Given a PEtab problem whose fit is poor, decide whether the cause is STRUCTURAL or "
    "NUMERICAL.\n"
    "- Structure: read get_model / get_model_structure. Ask whether the rate-law FORM is "
    "capable of the observed dynamics AT ALL, for ANY parameter value. Key facts: a "
    "first-order/linear term (k*X, or a bilinear k*X*Y) is unbounded and non-saturating; a "
    "rational law that divides by a state (Vmax*S/(Km+S), a Holling type-II response "
    "a*X/(1+a*h*X)) SATURATES; a Hill term with exponent n>1 (X^n/(K^n+X^n)) is sigmoidal "
    "and, in a feedback loop, is what makes SUSTAINED oscillations possible where a "
    "non-cooperative (n=1) loop only DAMPS to a steady state. Check conservation laws and "
    "whether the network can even reach the observed regime (a stable limit cycle vs a fixed "
    "point).\n"
    "- Numerics: read get_parameters (scales, bounds) and especially get_fitting_log — the "
    "pyPESTO/Fides multistart report. Did the starts converge to one optimum or scatter "
    "(non-identifiability)? Is a parameter pinned at a bound? Is the objective flat in some "
    "direction? Distinguish 'the optimizer failed' from 'the best possible fit for this "
    "structure is genuinely bad'.\n"
    "Use diff_versions/list_versions to see how the objective moved across edits. Comment on "
    "whether any proposed new parameters (a Hill exponent, a half-saturation constant, an "
    "inhibition constant) would be identifiable from the available data."
    + _REPORT_CONTRACT
)

SYSTEMS_SCIENTIST_PROMPT = (
    "You are a quantitative systems scientist (biochemistry, gene regulation, and "
    "population ecology). You judge whether a model's mechanism is how the system ACTUALLY "
    "works, not merely whether it fits. Read the mechanism (get_model / "
    "get_model_structure), what is measured (list_observables, get_measurements, "
    "list_conditions), and the parameter magnitudes (get_parameters).\n"
    "Ask mechanistic questions. What kind of process is each reaction — catalysis, binding, "
    "transcription/translation, degradation, predation, growth? Real mechanisms are almost "
    "always FINITE and therefore SATURATE: an enzyme has finite catalytic capacity "
    "(Michaelis-Menten v=Vmax*S/(Km+S); at high substrate it may even be inhibited); a "
    "predator has a finite handling time so its intake saturates with prey (a Holling "
    "type-II functional response, not the unbounded mass-action a*X*Y of textbook "
    "Lotka-Volterra); a transcription factor binds its operator COOPERATIVELY (a Hill "
    "response with exponent >1), which is what lets a repression loop oscillate rather than "
    "settle. Every species is produced AND removed (degradation/dilution). Read the DATA "
    "shape for fingerprints: a near-linear accumulation that later bends over is saturation; "
    "a rate that falls at high substrate is substrate inhibition; a sustained, "
    "roughly-constant-amplitude oscillation implies a limit cycle that a merely damped "
    "(non-cooperative / non-saturating) mechanism cannot sustain.\n"
    "Crucially, REJECT implausible fixes even if they would reduce residuals: inventing an "
    "unobserved species to absorb mass, or an autocatalytic term with no mechanism, is "
    "curve-fitting, not science. Prefer the parsimonious mechanism a domain expert would "
    "actually write down, reusing the model's existing species."
    + _REPORT_CONTRACT
)

SIGNAL_ENGINEER_PROMPT = (
    "You are a signal-processing engineer. You treat the measured and simulated time courses "
    "as signals and characterize the RESIDUALS — you do not hypothesize mechanisms. Use "
    "query_data (call list_data_tables first for the columns): the `simulations` table has "
    "one row per point with `measurement`, `simulation`, `residual`, `residual_weighted`, "
    "and `noise`. Compute your own residual breakdowns in SQL — overall and grouped by "
    "observable_id / condition_id / time, the worst-fit points "
    "(`ORDER BY abs(residual_weighted) DESC`), per-group bias (`avg(residual_weighted)`), "
    "and time-drift (`corr(time, residual_weighted)`).\n"
    "Report the STRUCTURE of the error. Is there a systematic bias (the model consistently "
    "over- or under-predicts)? Do residuals DRIFT with time (the model gets the "
    "dynamics/asymptote wrong, not just the level)? Is the mismatch in the CURVATURE or "
    "TIMING of the trajectory — data rises almost linearly early while the simulation is "
    "concave from t=0; a simulated peak shifted in time; the DATA keep oscillating with "
    "roughly constant amplitude while the SIMULATION's oscillation decays to a flat line "
    "(or vice versa); the response at a high-dose condition is lower than at a mid-dose one? "
    "Quantify where the fit is worst and whether the pattern is random (noise-level, fit is "
    "fine) or systematic (model-structure error). Localize the misfit precisely — which "
    "observable, which condition, which phase of the trajectory — and leave the WHY to the "
    "mathematician and systems scientist."
    + _REPORT_CONTRACT
)

# deepagents subagent specs. `subagent_type` values the orchestrator passes to the `task`
# tool are the `name`s here. Each names a curated tool subset (resolved from ALLOWED_TOOLS).
ANALYSTS = [
    {
        "name": "mathematician",
        "description": (
            "Mathematician: equation structure, dynamical-systems / CRN theory, and fit "
            "numerics. Ask it whether a misfit is a wrong rate-law FORM / broken conservation "
            "law / a regime the network cannot reach (structural) or non-identifiability / "
            "optimizer failure (numerical). Reads the fitting log."
        ),
        "prompt": MATHEMATICIAN_PROMPT,
        "tools": MATHEMATICIAN_TOOLS,
    },
    {
        "name": "systems_scientist",
        "description": (
            "Systems scientist (biochemistry / gene regulation / ecology): mechanistic "
            "plausibility. Ask it what kind of process a reaction is and whether a proposed "
            "fix is real mechanism (enzyme saturation, cooperative regulation, a saturating "
            "predator response, degradation, feedback) or a curve-fitting artifact to reject."
        ),
        "prompt": SYSTEMS_SCIENTIST_PROMPT,
        "tools": SYSTEMS_SCIENTIST_TOOLS,
    },
    {
        "name": "signal_engineer",
        "description": (
            "Signal-processing engineer: characterizes the residuals/trajectories. Ask it "
            "WHERE and HOW the fit fails — bias, time-drift, curvature/timing mismatch, a "
            "decaying vs sustained oscillation — without a mechanistic claim."
        ),
        "prompt": SIGNAL_ENGINEER_PROMPT,
        "tools": SIGNAL_ENGINEER_TOOLS,
    },
]


def _select_tools(tools: list) -> list:
    """Curate to ALLOWED_TOOLS. Warn (don't fail open) if expected tools are missing.

    Falling back to the full surface on a rename would silently dump the whole tool set
    into every agent's context — the opposite of the intent — so we keep the intersection
    and surface the drift instead.
    """
    available = {getattr(t, "name", None) for t in tools}
    missing = ALLOWED_TOOLS - available
    if missing:
        print(f"[agent] WARNING: expected tools not found on the server: {sorted(missing)} "
              "(tool names may have drifted)")
    selected = [t for t in tools if getattr(t, "name", None) in ALLOWED_TOOLS]
    if not selected:
        raise RuntimeError(
            "None of the curated ALLOWED_TOOLS matched the server's tools; refusing to fall "
            f"back to the full surface. Server tools: {sorted(n for n in available if n)}"
        )
    return selected


def _subagent_prompt_key() -> str:
    """The key deepagents' SubAgent spec uses for the persona prompt.

    deepagents renamed the subagent prompt field ``prompt`` -> ``system_prompt`` (mirroring
    the top-level ``create_deep_agent`` rename). Detect it from the ``SubAgent`` type so the
    panel works across releases; default to the current ``system_prompt``.
    """
    try:
        from deepagents.middleware.subagents import SubAgent

        keys = set(getattr(SubAgent, "__annotations__", {}) or {})
        if "system_prompt" not in keys and "prompt" in keys:
            return "prompt"
    except Exception:  # pragma: no cover - defensive
        pass
    return "system_prompt"


def _build_subagents(tools_by_name: dict[str, Any]) -> list[dict[str, Any]]:
    """Analyst specs with each persona's tool NAMES resolved to the server's tool OBJECTS.

    deepagents' ``SubAgent`` spec takes ``tools`` as a sequence of ``BaseTool``/callable
    objects (not name strings — older releases resolved names, current ones do not), so we
    map each analyst's curated name list to the actual loaded MCP tool objects, dropping any
    the server does not expose.
    """
    prompt_key = _subagent_prompt_key()
    subagents: list[dict[str, Any]] = []
    for spec in ANALYSTS:
        tools = [tools_by_name[n] for n in spec["tools"] if n in tools_by_name]
        dropped = [n for n in spec["tools"] if n not in tools_by_name]
        if dropped:
            print(f"[agent] WARNING: analyst {spec['name']!r} missing tools {dropped} "
                  "(dropped from its tool set)")
        subagents.append({
            "name": spec["name"],
            "description": spec["description"],
            prompt_key: spec["prompt"],
            "tools": tools,
        })
    return subagents


async def run_deepagents(
    session,
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    task: str,
    recursion_limit: int,
) -> dict[str, Any]:
    """Drive the orchestrator (+ analyst subagents) against ``session``'s tools.

    The model is served by OpenRouter via its OpenAI-compatible API (``base_url`` +
    ``api_key``). Returns ``{"stop_reason": ...}``. The orchestrator's messages, its
    ``task`` delegations, the analysts' private tool calls, and token usage are recorded by
    Phoenix tracing (set up by the runner), so this function just drives the loop to
    completion.
    """
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI

    # Adapt the MCP tools bound to this already-initialized session, then curate to the
    # union of the lead's + analysts' tools.
    tools = _select_tools(await load_mcp_tools(session))
    tools_by_name = {n: t for t in tools if (n := getattr(t, "name", None))}
    subagents = _build_subagents(tools_by_name)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        base_url=base_url,
        api_key=api_key,
    )

    # deepagents renamed the prompt kwarg between releases (instructions -> system_prompt).
    # Pass whichever the installed version accepts, plus the analyst subagents.
    params = inspect.signature(create_deep_agent).parameters
    prompt_kw = "system_prompt" if "system_prompt" in params else "instructions"
    kwargs: dict[str, Any] = {"model": llm, "tools": tools, prompt_kw: system_prompt}
    if "subagents" in params:
        kwargs["subagents"] = subagents
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `subagents` parameter; "
              "running a single agent without the analyst panel.")
    agent = create_deep_agent(**kwargs)

    stop_reason = "completed"
    # Drive the agent to completion. We consume the stream only to run the loop; the spans
    # Phoenix emits are the record of what happened.
    try:
        async for _state in agent.astream(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            pass
    except Exception as exc:
        # e.g. GraphRecursionError (step budget exhausted) or an OpenRouter error. Keep
        # whatever progress was made and let the judge assess the model as it stands,
        # rather than crashing the whole run.
        stop_reason = f"{type(exc).__name__}: {exc}"

    return {"stop_reason": stop_reason}
