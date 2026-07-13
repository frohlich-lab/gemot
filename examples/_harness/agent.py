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
  system actually works rather than a curve-fitting artifact.
* **signal_engineer** — the residuals and trajectories as *signals*. It reads the
  measured-vs-simulated points and the residual breakdown and reports *where* and *how* the
  fit fails (bias, drift, shape/timing mismatch) without claiming a mechanism.

The personas teach the analyst HOW to reason about a mis-specified dynamic model — its
lens, its tools, and the report it must return — but deliberately give NO hints about any
particular problem's answer (no named rate-law forms, no data fingerprints); the panel has
to diagnose the mechanism from the model and the data. The orchestrator prompt + task are
generic too (see ``scenario.DEFAULT_SYSTEM_PROMPT`` / ``DEFAULT_TASK``).

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

# Each analyst's tool set is kept as SMALL as its lens allows: every tool's JSON schema is
# re-sent to the model on every turn, so a lean set is the biggest lever on the (input-
# dominated) token cost. A tool the analyst can do without is dropped, not kept "just in
# case" — the union below is also all the panel ever sees.

# Mathematician: structural capability + estimation numerics. The reaction network (rate
# laws) + the parameter search space + the fit metrics + — uniquely — the pyPESTO/Fides log.
MATHEMATICIAN_TOOLS = [
    "get_model_structure",  # species, reactions, rate laws — the reaction network
    "get_parameters",       # scales, bounds — identifiability / conditioning
    "get_fit_summary",
    "get_fitting_log",      # pyPESTO/Fides log: multistart spread, convergence, bounds
]

# Systems scientist: is the mechanism real? The mechanism as written, what was measured
# (the readout + data shape), and whether the rate magnitudes are plausible.
SYSTEMS_SCIENTIST_TOOLS = [
    "get_model_structure",  # the mechanism as written
    "list_observables",     # what quantity is measured
    "get_measurements",     # the actual data / dynamic behaviour
    "get_parameters",       # are the rate magnitudes plausible?
]

# Signal-processing engineer: residuals as signals, entirely via SQL over the data.
SIGNAL_ENGINEER_TOOLS = [
    "list_data_tables",     # the queryable tables (simulations, residuals, ...) + columns
    "query_data",           # SQL over measured-vs-simulated points + residual breakdowns
    "get_fit_summary",
]

# The orchestrator ("modeling lead") orients, delegates analysis to the three analysts, and
# is the ONLY agent that edits the model or recovers. It keeps a MINIMAL surface: read the
# model + score, list versions (so a rollback is informed, not a guess), edit, batch, recover.
ORCHESTRATOR_TOOLS = [
    # orient / score
    "get_model_structure",
    "get_fit_summary",
    "list_versions",        # scored version history — so rollback targets the BEST version
    "get_errors",
    # edit the model (lead only)
    "add_reaction",
    "change_rate_law",
    "remove_reaction",
    # batch several related edits into one fitted version
    "begin_batch",
    "commit_batch",
    "abort_batch",
    # recover
    "rollback_to",
]

# The full set handed to create_deep_agent: the union of the lead's tools and every
# analyst's tools (deepagents resolves each subagent's `tools` names from this set).
ALLOWED_TOOLS = set(
    ORCHESTRATOR_TOOLS
    + MATHEMATICIAN_TOOLS
    + SYSTEMS_SCIENTIST_TOOLS
    + SIGNAL_ENGINEER_TOOLS
)

# Concise, panel-facing tool descriptions. The server's own docstrings are deliberately
# verbose (they double as the human/agent tool reference and drive the skill docs), but that
# full text is re-sent to the model on EVERY turn — the single largest fixed slice of the
# input-token cost. We override the description of each loaded MCP tool with a one-liner for
# the panel only; the server (and any other client) keeps its canonical docs. Anything not
# listed keeps (a truncated form of) its original description.
TOOL_DESCRIPTIONS = {
    "get_model_structure": "Structured model view: species, parameters, and reactions (id, reversibility, rate law).",
    "get_parameters": ("Parameter table: id, bounds, scale, nominalValue (the fitted value), "
                       "estimate flag. Fixed (estimate=0) rows include the observation-noise "
                       "parameters (sigma_*) and the parametric initial conditions (*_init) — "
                       "these belong to the noise / experimental model, NOT the kinetics, and "
                       "must never appear in a rate law."),
    "get_fit_summary": "Current fit metrics only (neg_log_likelihood, chi2, n_data_points) — a cheap score check.",
    "list_versions": ("List every model version with its kind, label, and fit metrics "
                      "(neg_log_likelihood, is_new_best) — use it to find the best-scoring "
                      "version to roll back to."),
    "get_fitting_log": "Tail of the pyPESTO/Fides multistart optimizer log (convergence, bound-pinning, ill-conditioning). Arg: tail_lines.",
    "list_observables": "The observables being fitted: id, formula, and noise model.",
    "get_measurements": "The measured data points. Optional args: condition_id, observable_id.",
    "list_data_tables": "List the SQL tables query_data exposes and their columns.",
    "query_data": ("Run a read-only DuckDB SQL query over the data. Tables: measurements, conditions, "
                   "observables, parameters, simulations (one row per point: measurement, simulation, "
                   "residual, residual_weighted, noise), optimization_starts, optimization_trace. "
                   "Args: sql, max_rows."),
    "get_errors": "Recent rejected operations / failed tool calls. Arg: limit.",
    "add_reaction": ("Add a reaction from an Antimony line (e.g. 'J: A -> B; k*A'); new symbols become "
                     "parameters/species. Args: reaction, justification, references?. Commits + auto-fits."),
    "change_rate_law": ("Replace a reaction's rate law; new symbols become parameters. Args: reaction_id, "
                        "rate_law, justification, references?. Commits + auto-fits."),
    "remove_reaction": "Remove a reaction and prune orphaned entities. Args: reaction_id, justification, references?. Commits + auto-fits.",
    "begin_batch": "Start a batch: later edits are staged (applied + logged, not fitted) until commit_batch. Arg: label.",
    "commit_batch": "Fit + simulate the staged edits once and snapshot them as one version. Args: label, n_starts?.",
    "abort_batch": "Discard the staged edits and restore the pre-batch version.",
    "rollback_to": "Restore a prior version as a new version. Args: version, justification?.",
}


def _trim_descriptions(tools: list) -> list:
    """Shorten each panel tool's description (in place) to cut per-turn input tokens.

    Uses the curated one-liner from TOOL_DESCRIPTIONS; for anything unlisted, truncates the
    server description to its first line. Best-effort — a tool whose description can't be set
    (e.g. a frozen model) is left as-is.
    """
    for t in tools:
        name = getattr(t, "name", None)
        short = TOOL_DESCRIPTIONS.get(name)
        if short is None:
            desc = getattr(t, "description", None)
            if isinstance(desc, str):
                short = desc.strip().split("\n", 1)[0]
        if short:
            try:
                t.description = short
            except Exception:  # pragma: no cover - frozen/rejected attribute
                pass
    return tools


# ---------------------------------------------------------------- analyst personas
# System prompts for the three analysts. Each is a focused expert that INSPECTS (it has no
# editing tools) and returns a concise written report ending in a concrete recommendation
# the lead can act on. They are independent: none sees the others' reports.
_REPORT_CONTRACT = (
    "\n\nYou cannot edit the model — you only inspect and advise. ALWAYS call at least one of "
    "your tools to inspect the actual problem before you conclude — do not answer from the "
    "task description alone. Then return a SHORT report: (1) what you found, (2) the single "
    "most likely explanation from your perspective, (3) a concrete recommendation for the "
    "modeling lead (name the operation and, if you can, the exact reaction and rate law). "
    "Flag explicitly if the evidence is ambiguous or outside your expertise. NEVER return an "
    "empty message: if the evidence is thin, still write the little you can conclude."
)

MATHEMATICIAN_PROMPT = (
    "You are a mathematician: ODE reaction-network and population dynamics, rate-law "
    "structure, and the numerics of parameter estimation. Given a PEtab problem that fits "
    "poorly, decide whether the cause is STRUCTURAL or NUMERICAL.\n"
    "- Structural: read get_model_structure and ask whether the rate-law FORM could produce "
    "the observed dynamics for ANY parameter values. Check conservation laws and whether the "
    "network can reach the observed regime.\n"
    "- Numerical: read get_parameters (scales, bounds) and get_fitting_log (the pyPESTO/Fides "
    "multistart). Did the starts converge, is a parameter pinned at a bound, is the objective "
    "flat in some direction (non-identifiability)? Distinguish 'the optimizer failed' from "
    "'the best fit for this structure is genuinely poor'. The parameter table also lists "
    "fixed observation-noise (sigma_*) and initial-condition (*_init) parameters — these are "
    "part of the noise/experimental model, not kinetic constants; never propose folding one "
    "into a rate law.\n"
    "Comment on whether any proposed new parameter would be identifiable from the data."
    + _REPORT_CONTRACT
)

SYSTEMS_SCIENTIST_PROMPT = (
    "You are a quantitative systems scientist (biochemistry, gene regulation, population "
    "ecology). You judge whether the model's mechanism reflects how the system actually "
    "works, not merely whether it fits. Read the mechanism (get_model_structure), what is "
    "measured (list_observables, get_measurements), and the "
    "parameter magnitudes (get_parameters). For each reaction, ask what real process it "
    "represents and whether its rate law is a faithful description of that process. Reject "
    "fixes that reduce residuals but are not real mechanism (e.g. inventing an unobserved "
    "species, or an ad-hoc term with no interpretation); prefer the most parsimonious "
    "mechanism a domain expert would actually write, reusing the model's existing species."
    + _REPORT_CONTRACT
)

SIGNAL_ENGINEER_PROMPT = (
    "You are a signal-processing engineer. Treat the measured and simulated time courses as "
    "signals and characterize the RESIDUALS — do not hypothesize mechanisms. Use query_data "
    "(call list_data_tables first for the columns): the `simulations` table has one row per "
    "point with `measurement`, `simulation`, `residual`, `residual_weighted`, and `noise`. "
    "Compute your own breakdowns in SQL — overall and grouped by observable_id / "
    "condition_id / time, the worst-fit points (`ORDER BY abs(residual_weighted) DESC`), "
    "per-group bias (`avg(residual_weighted)`), and time-trend "
    "(`corr(time, residual_weighted)`). Report the STRUCTURE of the error — a systematic "
    "bias, a drift with time, or a mismatch in the shape/timing of the trajectory — quantify "
    "where the fit is worst, and say whether the pattern looks like noise (the fit is fine) "
    "or systematic (a model-structure error). Localize the misfit; leave the WHY to the "
    "mathematician and systems scientist."
    + _REPORT_CONTRACT
)

# deepagents subagent specs. `subagent_type` values the orchestrator passes to the `task`
# tool are the `name`s here. Each names a curated tool subset (resolved from ALLOWED_TOOLS).
ANALYSTS = [
    {
        "name": "mathematician",
        "description": (
            "Mathematician: equation structure, dynamical-systems theory, and fit numerics. "
            "Ask whether a poor fit is structural (the rate-law form cannot produce the "
            "dynamics) or numerical (non-identifiability, optimizer failure). Reads the "
            "fitting log."
        ),
        "prompt": MATHEMATICIAN_PROMPT,
        "tools": MATHEMATICIAN_TOOLS,
    },
    {
        "name": "systems_scientist",
        "description": (
            "Systems scientist (biochemistry / gene regulation / ecology): mechanistic "
            "plausibility. Ask what real process a reaction represents and whether a proposed "
            "fix is a faithful mechanism or a curve-fitting artifact to reject."
        ),
        "prompt": SYSTEMS_SCIENTIST_PROMPT,
        "tools": SYSTEMS_SCIENTIST_TOOLS,
    },
    {
        "name": "signal_engineer",
        "description": (
            "Signal-processing engineer: characterizes the residuals/trajectories — where "
            "and how the fit fails (bias, drift, shape/timing mismatch) — with no mechanistic "
            "claim."
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


def _decision_tool():
    """A lead-only control tool that forces an explicit iterate / rollback / finish choice.

    The panel otherwise tends to make one edit and stop; requiring the lead to call this
    after each edit turns the run into an explicit model-building LOOP (and records the
    decision as a span). It doesn't drive control flow itself — the lead continues or stops
    by its own subsequent actions — it just makes the choice deliberate and auditable.
    """
    from langchain_core.tools import StructuredTool

    def make_decision(action: str, rationale: str = "") -> str:
        act = (action or "").strip().lower()
        print(f"[lead decision] {act or '?'} — {rationale}")
        guidance = {
            "iterate": "Recorded: iterate. Make the next single justified edit, then re-assess.",
            "rollback": "Recorded: rollback. Check list_versions and call rollback_to on the "
                        "BEST-scoring prior version (lowest neg_log_likelihood / is_new_best), "
                        "not just the previous one, then continue.",
            "finish": "Recorded: finish. Report the final neg_log_likelihood and exactly what "
                      "you changed, then stop.",
        }
        return guidance.get(act, "Unknown action — use one of: iterate, rollback, finish.")

    return StructuredTool.from_function(
        func=make_decision,
        name="make_decision",
        description=(
            "Record your control decision after assessing the current fit. `action` is one "
            "of: 'iterate' (the fit can still improve — do another edit), 'rollback' (the "
            "last edit did not help — revert it), or 'finish' (the fit is as good as it will "
            "get — stop). Give a one-line `rationale`. Call this once per round, after each "
            "edit, until you choose 'finish'."
        ),
    )


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
    tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
    tools_by_name = {n: t for t in tools if (n := getattr(t, "name", None))}
    subagents = _build_subagents(tools_by_name)
    # The lead-only decision tool: appended AFTER the subagent map is built so it goes to the
    # orchestrator (the full `tools` list) but not to any analyst.
    tools = [*tools, _decision_tool()]

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
