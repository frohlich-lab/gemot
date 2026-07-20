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
  theory, and the numerics of parameter estimation. From the measured dynamics it reasons
  about what reaction-network FORM could generate them; once a model exists it reads the
  equations, the parameter search space, and the pyPESTO/Fides *fitting log* to judge whether
  a misfit is structural (a rate-law FORM that cannot produce the observed dynamics, a broken
  conservation law) or numerical (non-identifiability, a start pinned at a bound).
* **systems_scientist** — mechanistic plausibility. From what is measured and how the data
  behaves it proposes the mechanism a domain expert would write (the species and their
  interactions); once a model exists it judges whether that mechanism is how the system
  actually works rather than a curve-fitting artifact.
* **signal_engineer** — the trajectories as *signals*. Before a model exists it characterizes
  the measured signal itself (shape, periodicity, timescales — the fingerprint the mechanism
  must reproduce); once a model is fitted it reads the measured-vs-simulated residual
  breakdown and reports *where* and *how* the fit fails (bias, drift, shape/timing mismatch)
  without claiming a mechanism.

The personas teach the analyst HOW to reason about building and refining a dynamic model —
its lens, its tools, and the report it must return — but deliberately give NO hints about any
particular problem's answer (no named rate-law forms, no data fingerprints); the panel has
to discover the mechanism from the data. The orchestrator prompt + task are generic too
(see ``scenario.DEFAULT_SYSTEM_PROMPT`` / ``DEFAULT_TASK``).

Each analyst is deliberately given only the tools its lens needs (see the ``*_TOOLS`` sets
below), so its context is not bloated with the full tool surface and its report stays in
character. The union of all analyst tools plus the lead's own editing/recovery tools is
handed to ``create_deep_agent`` (deepagents resolves each subagent's tool subset by name
from that union).

Observability is handled by **Arize Phoenix tracing** (see ``tracing.py``), which the
runner turns on before the agent starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
import time
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
    "get_model_structure",  # species, reactions, rate laws (empty until a model is built)
    "get_measurements",     # the DATA — dynamics/shape to reason about candidate structure
    "list_observables",     # what is measured
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
# is the ONLY agent that authors the model. It keeps a MINIMAL surface: read the model + score,
# list/inspect versions, author the whole model.
ORCHESTRATOR_TOOLS = [
    # orient — SMALL reads only. The measured DATA itself is never read into the lead's
    # context; the analysts characterize it and hand back summaries (see run_deepagents:
    # the lead's tool list is restricted to exactly this set).
    "describe_problem",     # model_initialized flag, sizes, current version — cheap orientation
    "get_model_structure",  # the current model (empty until built)
    "list_observables",     # what is measured (observable -> species) — needed to name species
    "list_conditions",      # per-condition inputs/doses set in the condition table (match by name)
    # score / diagnose (both COMPACT + bounded) / history
    "get_fit_summary",
    "get_fit_diagnosis",    # per-observable bias/chi2 + worst-N points — bounded residual read
    "get_identifiability",  # FIM SE/correlations + multistart spread — WHICH params to cut/merge
    "list_versions",        # scored version history — pick the BEST version to go back to
    "get_version",          # read a prior version's Antimony + params (to re-author it)
    "get_errors",
    # author the whole model in one step. The lead works exclusively by whole-model re-author:
    # across every traced run it never once used the incremental editors (add_reaction /
    # change_rate_law / remove_reaction) or batching, so those are omitted here to keep the
    # tool schema (re-sent every turn — the largest fixed input-token slice) lean. The server
    # still exposes them for other clients / scripted use. rollback_to is likewise omitted:
    # since set_model re-authors wholesale, "going back" = read the old model (get_version) and
    # set_model it again — no dedicated revert tool needed.
    "set_model",            # build the first model AND re-author it wholesale in one step
    # change a parameter's ESTIMATION in place — the prompt tells the lead to WIDEN a binding
    # bound (or switch scale) rather than re-typing a value the multistart fit ignores:
    "change_parameter_bounds",
    "change_parameter_scale",
    # NOTE: the UNBOUNDED data/diagnostic tools (get_measurements, query_data, get_fitting_log,
    # get_parameters, list_data_tables) are deliberately NOT here — they belong to the analyst
    # subagents, since their large results would pile up in the lead's re-sent history (the
    # input-token cost). The lead instead has get_fit_diagnosis (a COMPACT, size-bounded residual
    # read) for cheap self-diagnosis, and DELEGATES the deep dives — full residual SQL, the
    # optimizer log, raw data characterization — to the analysts via `task`.
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
    "set_model": ("Author the WHOLE model from ONE Antimony string (compartments, species with "
                  "initial conditions, parameters, reactions). BUILDS the first model when none "
                  "exists (model_initialized: false) AND re-authors/replaces an existing one — never "
                  "refused. Use for the initial build and for any coupled/symmetric or multi-reaction "
                  "STRUCTURAL change in one coherent step; reconciles + fits on commit, reverts on "
                  "failure. Warns if a previously-estimated parameter is dropped. Args: antimony, "
                  "justification, references?."),
    "describe_problem": "Orientation: whether a model exists (model_initialized), table/entity counts, current version, batch state. Cheap.",
    "get_model_structure": "Structured model view: species, parameters, and reactions (id, reversibility, rate law) — empty until a model is built.",
    "get_parameters": ("Parameter table: id, bounds, scale, nominalValue (the fitted value), "
                       "estimate flag. Fixed (estimate=0) rows include the observation-noise "
                       "parameters (sigma_*) and the parametric initial conditions (*_init) — "
                       "these belong to the noise / experimental model, NOT the kinetics, and "
                       "must never appear in a rate law."),
    "get_fit_summary": "Current fit metrics only (neg_log_likelihood, chi2, n_data_points) — a cheap score check.",
    "get_fit_diagnosis": ("COMPACT residual diagnosis (bounded): per-observable weighted chi2 + "
                          "mean_weighted_residual (bias) + worst-N points (measurement vs "
                          "simulation). Shows WHERE/HOW the fit is off cheaply. Arg: worst_n. For a "
                          "full custom breakdown or the optimizer log, delegate to an analyst."),
    "get_identifiability": ("COMPACT identifiability diagnosis (bounded): per-parameter standard "
                            "error + fold_uncertainty, marginal & partial correlations, near-flat "
                            "FIM eigen-directions (non-identifiable parameter COMBINATIONS), and "
                            "multistart spread — with a ranked `candidates` list. Tells you WHICH "
                            "estimated parameters are superfluous when the model is too complex "
                            "(high bic / many params): fix, drop, or MERGE a collinear pair. Use it "
                            "BEFORE adding parameters and to prune toward parsimony. Arg: top_n."),
    "list_versions": ("List every model version with its kind, label, and fit metrics "
                      "(neg_log_likelihood, bic, is_new_best) — use it to find the best-scoring "
                      "version, then get_version to read its model and re-author with set_model."),
    "get_version": ("Read a prior version's Antimony model + parameter table. Use it to go back "
                    "to an earlier model: get_version(v) then re-author it with set_model. Arg: version."),
    "get_fitting_log": "Tail of the pyPESTO/Fides multistart optimizer log (convergence, bound-pinning, ill-conditioning). Arg: tail_lines.",
    "list_observables": "The observables being fitted: id, formula, and noise model.",
    "get_measurements": ("Measurement data. Default returns a COMPACT SUMMARY (per observable×"
                         "condition: n, time span, value min/max/mean, preview points); pass "
                         "summarize=False for all raw points. Optional: condition_id, observable_id."),
    "list_data_tables": "List the SQL tables query_data exposes and their columns.",
    "query_data": ("Run a read-only DuckDB SQL query over the data. Tables: measurements, conditions, "
                   "observables, parameters, simulations (one row per point: measurement, simulation, "
                   "residual, residual_weighted, noise), optimization_starts, optimization_trace. "
                   "Prefer aggregations; rows capped at max_rows (default 50 — raise it only if you "
                   "truly need many rows). Args: sql, max_rows."),
    "get_errors": "Recent rejected operations / failed tool calls. Arg: limit.",
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
    "structure, and the numerics of parameter estimation. The problem may ship WITHOUT a "
    "model yet — help BUILD the right structure, then judge it.\n"
    "- Structure: from the measured dynamics (get_measurements, list_observables) reason about "
    "what reaction-network FORM could GENERATE them — the state dimension, which rate laws are "
    "linear vs saturating vs cooperative, the conservation laws implied. Once a model exists "
    "(get_model_structure), ask whether its rate-law FORM can produce the observed dynamics "
    "for ANY parameter values.\n"
    "- Numerics: read get_parameters (scales, bounds) and, once a fit exists, get_fit_summary "
    "and get_fitting_log (the pyPESTO/Fides multistart). Did the starts converge, is a "
    "parameter pinned at a bound, is the objective flat in some direction "
    "(non-identifiability)? Distinguish 'the optimizer failed' from 'this structure genuinely "
    "cannot fit'. The parameter table also lists fixed observation-noise (sigma_*) and "
    "initial-condition (*_init) parameters — part of the noise/experimental model, not kinetic "
    "constants; never fold one into a rate law.\n"
    "Comment on whether the free parameters (proposed or present) are identifiable from the data."
    + _REPORT_CONTRACT
)

SYSTEMS_SCIENTIST_PROMPT = (
    "You are a quantitative systems scientist (biochemistry, gene regulation, population "
    "ecology). From what is measured (list_observables, get_measurements) and the data's "
    "qualitative behaviour, propose the mechanism a domain expert would actually write for "
    "this system: the species, the interactions between them, and the rate-law FORM of each "
    "process. If a model already exists (get_model_structure), judge instead whether each "
    "reaction faithfully describes a real process rather than a curve-fitting artifact (an "
    "invented unobserved species, an ad-hoc term with no interpretation), and check the rate "
    "magnitudes (get_parameters) are physically plausible. Prefer the most parsimonious "
    "mechanism that explains the data, reusing the measured species."
    + _REPORT_CONTRACT
)

SIGNAL_ENGINEER_PROMPT = (
    "You are a signal-processing engineer. Treat the time courses as signals — do not "
    "hypothesize mechanisms. Call list_data_tables FIRST for the available tables/columns, "
    "then use query_data. Before a model is fitted, characterize the MEASURED signal itself, "
    "per observable_id / condition_id: its shape (monotonic rise, saturation, decay, sustained "
    "oscillation — with its period — or damping), timescales, and amplitudes — the qualitative "
    "fingerprint the mechanism must reproduce. Once a model is fitted, the `simulations` table "
    "has one row per point with `measurement`, `simulation`, `residual`, `residual_weighted`, "
    "and `noise`: compute breakdowns in SQL — grouped by observable_id / condition_id / time, "
    "the worst-fit points (`ORDER BY abs(residual_weighted) DESC`), per-group bias "
    "(`avg(residual_weighted)`), and time-trend (`corr(time, residual_weighted)`) — and say "
    "whether the residuals look like noise (the fit is adequate) or systematic (the structure "
    "is still wrong). Localize; leave the WHY to the mathematician and systems scientist."
    + _REPORT_CONTRACT
)

# deepagents subagent specs. `subagent_type` values the orchestrator passes to the `task`
# tool are the `name`s here. Each names a curated tool subset (resolved from ALLOWED_TOOLS).
ANALYSTS = [
    {
        "name": "mathematician",
        "description": (
            "Mathematician: equation structure, dynamical-systems theory, and fit numerics. "
            "Ask what reaction-network structure the measured dynamics imply, and — once a "
            "model exists — whether a poor fit is structural (the rate-law form cannot produce "
            "the dynamics) or numerical (non-identifiability, optimizer failure). Reads the "
            "fitting log."
        ),
        "prompt": MATHEMATICIAN_PROMPT,
        "tools": MATHEMATICIAN_TOOLS,
    },
    {
        "name": "systems_scientist",
        "description": (
            "Systems scientist (biochemistry / gene regulation / ecology): mechanistic "
            "plausibility. Ask what mechanism the measured system has — the species and their "
            "interactions — and whether a reaction is a faithful process or a curve-fitting "
            "artifact to reject."
        ),
        "prompt": SYSTEMS_SCIENTIST_PROMPT,
        "tools": SYSTEMS_SCIENTIST_TOOLS,
    },
    {
        "name": "signal_engineer",
        "description": (
            "Signal-processing engineer: characterizes the measured signal (shape, "
            "periodicity, timescales) and — once a model is fitted — the residuals (bias, "
            "drift, shape/timing mismatch), with no mechanistic claim."
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


_MAX_FINISH_REFUSALS = 2   # push back on finishing with NO valid fit this many times, then relent


async def _current_fit(session) -> dict[str, Any]:
    """Fetch the CURRENT HEAD's fit metrics from the server (ground truth, not the agent's
    self-report). Returns {} if the metrics can't be read — the finish check then sees no
    valid fit and fails CLOSED: it pushes back on ``finish`` (bounded by
    ``_MAX_FINISH_REFUSALS``) rather than letting the run end with no scored model."""
    try:
        res = await session.call_tool("get_fit_summary", {})
        sc = getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            return sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


def _decision_tool(session):
    """A lead-only control tool for the two non-default decisions: ``rollback`` and ``finish``.

    Plain iteration needs no call — to keep going the lead simply makes the next ``set_model``
    edit — so the tool exists only to record a rollback or to gate a finish.

    The panel otherwise tends to make one edit and stop, so ``finish`` is pushed back — but
    ONLY while there is no valid fit to finish on: no model has been built yet, or the fit's
    objective is non-finite. BIC is the success target and there is no BIC without a fit, so
    that (and only that) is a genuine reason to keep going. The push-back is bounded — after
    ``_MAX_FINISH_REFUSALS`` attempts it RELENTS so a genuinely-stuck run ends cleanly instead
    of looping to the recursion limit.

    It deliberately does NOT gate on the χ² goodness-of-fit (chi2_gof_pvalue / reduced_chi2).
    Whenever the observable noise sigmas are estimated (as in the issue-#54 benchmarks — though
    not every problem frees them), χ² pins near the data-point count by construction and its
    p-value is uninformative, so gating ``finish`` on p ≥ 0.05 made the panel abandon its
    best-BIC model to chase an unreachable target (issue #54). Where χ² cannot be computed at
    all, the tool SAYS so rather than surfacing it as a fit rejection.
    """
    from langchain_core.tools import StructuredTool

    _guidance = {
        "rollback": "Recorded: rollback. Find the BEST-scoring prior version with list_versions "
                    "(LOWEST bic / is_new_best, not just the previous one), read it with "
                    "get_version, and re-author it with set_model — then continue from there.",
        "finish": "Recorded: finish. Report the final neg_log_likelihood and bic and exactly "
                  "what you changed, then stop.",
    }
    state = {"finish_refusals": 0}  # bounded push-back on finishing with no valid fit

    async def make_decision(action: str, rationale: str = "") -> str:
        act = (action or "").strip().lower()
        print(f"[lead decision] {act or '?'} — {rationale}")
        if act == "finish":
            fit = await _current_fit(session)
            nll = fit.get("neg_log_likelihood")
            bic = fit.get("bic")
            # Require a finite BIC — the score the harness actually finishes on. A finite nll is
            # necessary but not sufficient: get_fit_summary reports bic=None when n_data_points/k
            # are incomplete, and finishing then would report "bic=n/a" while the refusal text
            # promises "there is no bic to report". Gate on the score that must exist.
            def _finite(x):
                return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            has_fit = _finite(nll) and _finite(bic)
            # Block finish ONLY when there is no valid fit to finish on (nothing built, a
            # non-finite objective, or no computable bic) — a genuine failure, not a
            # goodness-of-fit judgement. We do NOT gate on chi2_gof_pvalue/reduced_chi2 (see the
            # docstring): they are uninformative when the noise sigmas are estimated, and chasing
            # p ≥ 0.05 costs the best-BIC model.
            if not has_fit:
                if state["finish_refusals"] < _MAX_FINISH_REFUSALS:
                    state["finish_refusals"] += 1
                    print(f"[finish BLOCKED {state['finish_refusals']}/{_MAX_FINISH_REFUSALS}] "
                          "no valid fit (no model built, non-finite objective, or no bic)")
                    return (
                        f"FINISH REFUSED ({state['finish_refusals']}/{_MAX_FINISH_REFUSALS}) — "
                        "there is no valid fit to finish on: no model has been built yet, or the "
                        "fit has no finite bic to report (a non-finite objective, or an incomplete "
                        "n/k). Build a model that simulates and fits, then finish on its bic."
                    )
                # exhausted the push-back: relent so the run ends cleanly, not at the step limit
                print(f"[finish ALLOWED] after {state['finish_refusals']} refusals; still no "
                      "valid fit")
                return (
                    f"Recorded: finish — NOTE there is STILL no valid fit after "
                    f"{state['finish_refusals']} further attempts. State plainly that no fitted "
                    "model could be built, then stop."
                )
            # Valid fit -> finish is allowed. BIC is the target; χ² is NOT a gate. When the χ²
            # metrics are absent, say so (and why) rather than surfacing it as a fit rejection.
            # goodness_of_fit leaves them None when the degrees of freedom n - k are <= 0 or chi2
            # itself is missing — NOT because sigmas are estimated (with estimated sigmas χ² is
            # still computed, just uninformative, which the prompt/docstrings cover separately).
            chi2_note = ""
            if fit.get("chi2_gof_pvalue") is None and fit.get("reduced_chi2") is None:
                chi2_note = (" (χ² goodness-of-fit is not reported here — n - k <= 0 or chi2 is "
                             "unavailable, so it could not be computed; this is NOT a fit rejection.)")
            return (
                f"Recorded: finish — final bic={bic:.6g}, neg_log_likelihood={nll:.6g}."
                f"{chi2_note} Report the bic, the neg_log_likelihood, and exactly what you "
                "changed, then stop."
            )
        return _guidance.get(act, "Unknown action — use one of: rollback, finish.")

    return StructuredTool.from_function(
        coroutine=make_decision,
        name="make_decision",
        description=(
            "Record a control decision after reading the CURRENT HEAD's fit metrics "
            "(bic / neg_log_likelihood / is_new_best — never an earlier version's). You do NOT "
            "call this to keep iterating — to iterate, just make the next set_model edit. Call it "
            "only for `action`: 'rollback' (the last change didn't help — re-author the LOWEST-bic "
            "prior version via get_version + set_model) or 'finish' (residuals show no systematic "
            "structure AND bic has bottomed out). BIC is the ONLY success target; chi2_gof_pvalue "
            "/ reduced_chi2 are UNINFORMATIVE when the noise sigmas are estimated — do NOT keep "
            "iterating just to chase chi2_gof_pvalue ≥ 0.05. 'finish' is refused only when there "
            "is no valid fit yet (no model built). Prefer the FEWEST free parameters that fit "
            "(lower bic wins). Give a one-line `rationale`."
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


# deepagents injects a set of built-in filesystem + todo tools (ls/glob/read_file/write_file/
# edit_file/grep/execute/write_todos) into every agent, on top of the `tools` we pass. This
# task has no filesystem — the real tools are the MCP tools — so those built-ins are pure
# overhead: their JSON schemas are re-sent on every turn (the input-token cost this module
# works hard to keep lean, see _trim_descriptions/TOOL_DESCRIPTIONS), and each call the model
# wastes on them also burns a recursion step. `task` (subagent delegation) is NEVER excluded —
# the panel depends on it. We attach this filter to the LEAD only: neither the per-model
# `HarnessProfile.excluded_tools` route nor a per-subagent `middleware` list reliably reaches
# deepagents 0.6.12's declarative analyst subagents (both were verified to still leak FS calls
# there), so the analysts keep their built-ins for now — the lead is the heavy tool-caller and
# is the one worth trimming. Filtering the subagents cleanly needs an upstream deepagents fix.
_BUILTIN_TOOLS_TO_HIDE = frozenset({
    "write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
})


def _builtin_tool_filter():
    """A fresh middleware instance that strips ``_BUILTIN_TOOLS_TO_HIDE`` from the model call.

    Mirrors deepagents' own (private) ``_ToolExclusionMiddleware``: it removes the named tools
    from ``request.tools`` before the model sees them, in BOTH the sync and async paths (the
    deepagents driver runs via ``astream``, so ``awrap_model_call`` is the one that fires).
    Defined lazily so importing this module doesn't require langchain to be installed — only
    the deepagents driver, which pulls langchain in, ever calls this.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    def _name(tool) -> str | None:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    class _DropBuiltinTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            request = request.override(
                tools=[t for t in request.tools if _name(t) not in _BUILTIN_TOOLS_TO_HIDE])
            return handler(request)

        async def awrap_model_call(self, request, handler):
            request = request.override(
                tools=[t for t in request.tools if _name(t) not in _BUILTIN_TOOLS_TO_HIDE])
            return await handler(request)

    return _DropBuiltinTools()


# --- repeated-tool-error circuit breaker --------------------------------------------------
# A broken tool used to be re-proposed cumulatively hundreds of times across a run (a
# trace-review swarm saw one query_data call re-proposed ~798× — up to ~100 pending copies in
# a single step) while the agent learned nothing from the identical failures. This breaker
# counts every tool failure and, after N CONSECUTIVE IDENTICAL errors from the same tool,
# DISABLES that tool for the rest of the run: further calls are short-circuited with a terminal
# message pointing at the working alternative, so the agent stops hammering a dead tool and
# moves on. A SUCCESS resets a tool's streak; DIFFERENT errors reset it too (the agent is still
# exploring), so only a genuinely-stuck tool is cut off.
#
# It works at the TOOL-OBJECT layer (wrapping each loaded tool's coroutine), not as agent
# middleware, deliberately: the heavy data/diagnostic tools (query_data, get_fitting_log, …)
# run INSIDE the analyst subagents, and deepagents 0.6.12's declarative subagents don't
# reliably receive a middleware list (see _builtin_tool_filter). Wrapping the tool object makes
# the breaker travel WITH the tool into whichever agent (lead or analyst) calls it.
_TOOL_ERROR_DISABLE_THRESHOLD = 3   # consecutive identical errors from one tool before it is cut

# Where to send the agent when a specific tool is disabled (its result already carries the
# server's own hint; this is the panel-facing fallback). Anything unlisted gets the generic one.
_DISABLED_TOOL_HINTS = {
    "query_data": ("Use `get_measurements` for the raw measurement data, or `list_data_tables` "
                   "for the available tables and their columns."),
}
_DISABLED_TOOL_HINT_DEFAULT = (
    "Use a different tool (or `get_errors` to review the failure) and continue without it."
)


def _new_breaker_state() -> dict[str, Any]:
    """Mutable, run-scoped state shared by the wrapped tools and the lead's tool filter."""
    return {"disabled": set(), "n_tool_errors": 0, "streaks": {}, "events": []}


def _breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """The terminal content block returned in place of a disabled tool's result."""
    return [{"type": "text", "text": (
        f"TOOL DISABLED: `{name}` failed repeatedly with the same error and is disabled for the "
        f"rest of this run. Do NOT call it again — {hint}"
    )}]


def _result_error(content: Any, artifact: Any) -> str | None:
    """Return the error string if a (content, artifact) tool result is a FAILURE, else None.

    The petab-mcp server returns a failed read/diagnostic tool as a normal result carrying
    ``{"ok": false, "error": ...}`` (not an MCP protocol error), which langchain-mcp-adapters
    exposes in the artifact's ``structured_content`` and in the text content. A rejected
    *operation* (``{"committed": false}`` — e.g. a lint failure the agent should fix and retry)
    is deliberately NOT treated as a breaker error: that is the normal edit/retry feedback loop.
    """
    if isinstance(artifact, dict):
        sc = artifact.get("structured_content")
        if isinstance(sc, dict) and sc.get("ok") is False:
            return str(sc.get("error") or "tool error")
        if artifact.get("isError") or artifact.get("is_error"):
            return "tool error"
    # Fallback: concatenate any text content blocks and look for the {"ok": false} shape.
    text = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text") or "")
            elif isinstance(block, str):
                text += block
    elif isinstance(content, str):
        text = content
    try:
        import json
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("ok") is False:
        return str(obj.get("error") or "tool error")
    return None


def _record_error(state: dict[str, Any], name: str, err: str) -> bool:
    """Count a tool error; return True iff it TRIPS the breaker (disables the tool now)."""
    state["n_tool_errors"] += 1
    sig = err[:200]
    streak = state["streaks"].get(name)
    if streak and streak["sig"] == sig:
        streak["count"] += 1
    else:
        streak = {"sig": sig, "count": 1}
        state["streaks"][name] = streak
    if streak["count"] >= _TOOL_ERROR_DISABLE_THRESHOLD and name not in state["disabled"]:
        state["disabled"].add(name)
        msg = (f"disabled `{name}` after {streak['count']} consecutive identical errors: "
               f"{sig}")
        state["events"].append(msg)
        print(f"[circuit-breaker] {msg}")
        return True
    return False


def _wrap_tools_with_circuit_breaker(tools: list, state: dict[str, Any]) -> list:
    """Wrap each tool's coroutine so repeated identical failures disable it (see above).

    Mutates the tool objects in place (and returns the list) so the breaker rides along into
    the analyst subagents, which hold these same objects. Best-effort: a tool without a
    coroutine is left untouched.
    """
    import inspect
    from functools import wraps

    for tool in tools:
        name = getattr(tool, "name", None)
        orig = getattr(tool, "coroutine", None)
        if name is None or orig is None:
            continue
        hint = _DISABLED_TOOL_HINTS.get(name, _DISABLED_TOOL_HINT_DEFAULT)

        def _make(name: str, orig, hint: str):
            @wraps(orig)
            async def _breaker(runtime=None, **arguments):
                if name in state["disabled"]:
                    return (_breaker_message(name, hint), None)  # short-circuit, no server hit
                try:
                    result = await orig(runtime=runtime, **arguments)
                except Exception as exc:  # a raised tool error (e.g. MCP isError -> ToolException)
                    if _record_error(state, name, f"{type(exc).__name__}: {exc}"):
                        return (_breaker_message(name, hint), None)
                    raise
                content, artifact = result if isinstance(result, tuple) else (result, None)
                err = _result_error(content, artifact)
                if err is None:
                    state["streaks"].pop(name, None)  # a success clears the tool's error streak
                    return result
                if _record_error(state, name, err) and isinstance(content, list):
                    return (content + _breaker_message(name, hint), artifact)
                return result

            try:  # keep the original signature so langchain still injects `runtime` correctly
                _breaker.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _breaker

        tool.coroutine = _make(name, orig, hint)
    return tools


def _disabled_tool_filter(state: dict[str, Any]):
    """Lead-only middleware: hide a disabled tool from the model so it stops being re-proposed.

    Complements the coroutine wrapper (the wrapper is what actually reaches the analyst
    subagents): for the LEAD's own tools this also drops the tool's schema from the re-sent
    request once it is disabled, saving tokens and removing the temptation to call it again.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    def _name(tool) -> str | None:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _keep(request):
        return [t for t in request.tools if _name(t) not in state["disabled"]]

    class _DropDisabledTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request.override(tools=_keep(request)))

        async def awrap_model_call(self, request, handler):
            return await handler(request.override(tools=_keep(request)))

    return _DropDisabledTools()


# --- guards on the (input-dominated, occasionally-hanging) LLM loop -----------------------
# Transient provider failures (connection drops, malformed/non-JSON bodies, 5xx/429, hung
# calls) are common on OpenRouter and used to abandon the whole run. Now they RETRY instead of
# throwing the run away, in two layers:
#   * per-CALL: the OpenAI client's own ``max_retries`` retries connection errors, request
#     timeouts, and 5xx/429 (honouring Retry-After) transparently, so those never reach the loop.
#   * per-RUN resume: the agent is compiled with a LangGraph checkpointer, so anything that still
#     escapes — a malformed non-JSON body the client won't retry, exhausted call-retries, or a
#     STALL the client timeout can't catch — RESUMES ``astream`` from the last checkpoint instead
#     of aborting, so every committed model version and the conversation state are preserved.
_LLM_TIMEOUT_S = 120.0
_LLM_MAX_RETRIES = 4           # OpenAI-client retries for connection / timeout / 5xx / 429
_RUN_RESUME_ATTEMPTS = 3       # resume the graph from the last checkpoint after a stall/transient
# Harness-level stall watchdog on the agent loop. The client `timeout` above does NOT reliably
# catch a mid-stream provider stall (observed: a 24-min no-progress hang that the 120s timeout
# never fired on). astream emits once per graph super-step (a lead LLM call, a tool, or a whole
# analyst sub-agent), so if NO step completes within this window the loop is wedged — we cancel
# it and RESUME from the last checkpoint (or, once resume attempts run out, end gracefully).
# This is only the FLOOR for the non-fit steps: run_deepagents raises the effective window to the
# full multistart fit budget (fit_budget.py) for the run, since a lead commit's fit is one long
# super-step that legitimately streams nothing back for minutes.
_STEP_STALL_TIMEOUT_S = 600.0

# History trimming for the LEAD. This workload is ~90% input tokens, dominated by tool RESULTS
# (analyst reports, set_model/commit payloads, get_measurements/query_data dumps) re-sent on
# every turn — which ballooned a run to ~1.1M tokens and slow/hanging calls. The panel re-reads
# live state each round (get_fit_summary / get_model_structure), so stale tool outputs in the
# history are pure overhead: once the lead's context exceeds the trigger we CLEAR the oldest
# tool results (keeping the most-recent few verbatim + all tool CALLS, so tool-call/result
# pairing stays valid), replacing them with a small placeholder. No extra LLM calls.
# Relaxed from an aggressive 45k/keep-3: at that setting the lead lost the record of what its
# earlier attempts fit to and flip-flopped between structures instead of converging. Keep much
# more history (higher trigger + more recent tool results retained) so it can learn from its
# own attempt log; still bounded well under the model's context window, and the per-request
# timeout guard backstops any slow call so relaxing this can't reinstate the old ~1M-token hang.
_CONTEXT_CLEAR_TRIGGER = 90_000     # tokens; clear old tool results once the context exceeds this
_CONTEXT_KEEP_TOOL_RESULTS = 8      # keep this many most-recent tool results verbatim


def _history_trimmer():
    """Context-editing middleware (lead-only) that clears stale tool results — see above."""
    from langchain.agents.middleware import ContextEditingMiddleware
    from langchain.agents.middleware.context_editing import ClearToolUsesEdit

    return ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=_CONTEXT_CLEAR_TRIGGER,
                                 keep=_CONTEXT_KEEP_TOOL_RESULTS)],
        token_count_method="approximate",
    )


async def run_deepagents(
    session,
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    task: str,
    recursion_limit: int,
    temperature: float = 0.0,
    n_starts: int = 1,
    max_wall_seconds: float | None = None,
    progress_probe: Any | None = None,
) -> dict[str, Any]:
    """Drive the orchestrator (+ analyst subagents) against ``session``'s tools.

    ``max_wall_seconds`` (optional) is a self-imposed total-runtime budget, checked between
    graph super-steps: once exceeded, the loop stops gracefully (like running out of resume
    attempts) so ``_judge`` can score whatever is currently committed, INSTEAD of racing an
    outer resource limit (e.g. a Slurm ``--time`` allocation) that would hard-kill the process
    with no chance to write a result at all. A sweep should size this comfortably BELOW its
    Slurm time limit (see ``slurm/make_sweep.py``'s ``--max-wall-seconds`` / ``--time``) so this
    fires first. ``None`` (the default) means no self-imposed limit — only the per-step stall
    watchdog and the outer scheduler's own limit apply, as before.

    ``progress_probe`` (optional) is an un-timed ``await probe(tool, args)`` handle used to poll
    the server's ``get_progress`` between super-steps: when the early-stop chain reports
    ``should_stop`` (the non-improving-streak cap, or a plateau once the best model is adequate),
    the loop stops at the next clean boundary — the deterministic replacement for wandering to the
    recursion limit. It is passed un-timed so these internal polls don't inflate the tool clock /
    ``tool_calls`` metric. ``None`` disables the check (the loop runs as before).

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
    # union of the lead's + analysts' tools. This pool is only used to resolve each analyst's
    # tool objects; it is NOT what the lead gets (see lead_tools below).
    all_tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
    # Wrap the tool objects with the repeated-error circuit breaker BEFORE they are handed to
    # the lead and the analyst subagents (both reference these same objects), so a tool that
    # keeps failing identically is disabled for the run instead of being re-proposed endlessly.
    breaker_state = _new_breaker_state()
    _wrap_tools_with_circuit_breaker(all_tools, breaker_state)
    tools_by_name = {n: t for t in all_tools if (n := getattr(t, "name", None))}
    subagents = _build_subagents(tools_by_name)  # each analyst carries its own tool OBJECTS
    # The LEAD gets ONLY the orchestrator tools (+ its decision tool) — NOT the analysts'
    # data/diagnostic tools (query_data, get_fitting_log, get_measurements, ...). Those heavy
    # reads run INSIDE analyst subagents, whose verbose results live in their own transient
    # contexts and never enter the lead's every-turn re-sent history — which is where this
    # input-dominated workload's token cost accumulates (the lead re-sends its whole history on
    # each of its 60-115 turns). The lead diagnoses by DELEGATING to the analysts, not by
    # querying data itself. deepagents' subagents hold their own resolved tool objects, so a
    # restricted top-level tool list does not starve them.
    lead_tools = [tools_by_name[n] for n in ORCHESTRATOR_TOOLS if n in tools_by_name]
    lead_tools = [*lead_tools, _decision_tool(session)]

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_S,          # fail a hung provider call fast (was unbounded)
        max_retries=_LLM_MAX_RETRIES,
        # Best-effort prompt-cache hint: asks the provider to route repeat requests to the
        # same backend so its prefix cache can hit (prompt tokens dominate this workload).
        # Whether it takes effect is provider-dependent (OpenRouter -> the served model) and
        # not guaranteed; it is harmless when ignored.
        model_kwargs={"prompt_cache_key": "petab-mcp-agent"},
    )
    # Transient provider errors that should RESUME the run (below) rather than end it, when they
    # escape the OpenAI client's own retries — e.g. a malformed non-JSON body the client won't
    # retry. Non-transient errors (bad request, recursion-limit) are deliberately excluded.
    from json import JSONDecodeError

    from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                        RateLimitError)
    _TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError,
                         RateLimitError, JSONDecodeError)

    # deepagents renamed the prompt kwarg between releases (instructions -> system_prompt).
    # Pass whichever the installed version accepts, plus the analyst subagents.
    params = inspect.signature(create_deep_agent).parameters
    prompt_kw = "system_prompt" if "system_prompt" in params else "instructions"
    kwargs: dict[str, Any] = {"model": llm, "tools": lead_tools, prompt_kw: system_prompt}
    if "subagents" in params:
        kwargs["subagents"] = subagents
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `subagents` parameter; "
              "running a single agent without the analyst panel.")
    # Lead middleware: drop deepagents' built-in FS/todo tools (the analysts can't be reliably
    # filtered in this deepagents version — see _builtin_tool_filter; `task` is untouched), and
    # trim the lead's re-sent tool-result history so the (input-dominated) context stays bounded
    # instead of ballooning to ~1M tokens and hanging.
    if "middleware" in params:
        kwargs["middleware"] = [
            _builtin_tool_filter(), _disabled_tool_filter(breaker_state), _history_trimmer(),
        ]
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `middleware` parameter; "
              "the lead's built-in FS/todo tools + full history will remain.")
    # A checkpointer lets us RESUME the graph from its last completed super-step after a stall or
    # exhausted call-retry (below), instead of discarding the run. In-process MemorySaver is
    # enough (one run per process). Degrade gracefully if the graph can't take one.
    checkpointer = None
    if "checkpointer" in params:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            kwargs["checkpointer"] = checkpointer
        except Exception as exc:  # pragma: no cover - resume is a best-effort enhancement
            print(f"[agent] no checkpointer ({type(exc).__name__}); a stall will end the run "
                  "instead of resuming.")
    agent = create_deep_agent(**kwargs)

    stop_reason = "completed"
    # Drive the agent to completion, iterating manually so a per-step STALL WATCHDOG can fire:
    # astream emits once per graph super-step, so if `__anext__` doesn't return within the stall
    # window the current step is wedged. On a stall — or a call-retry that ran out (below) — we
    # RESUME from the last checkpoint (astream with input=None continues pending work) rather than
    # abandoning the run, up to _RUN_RESUME_ATTEMPTS times. A non-transient error (e.g.
    # GraphRecursionError = step budget exhausted) is NOT retried; the judge assesses the model as
    # it stands. We consume the stream only to drive the loop; Phoenix spans are the record.
    # The longest single super-step is a lead commit whose set_model runs an n_starts multistart
    # fit (astream emits nothing until it returns), so the stall window must exceed one whole fit
    # or a slow-but-healthy fit trips the watchdog. Size it to the same fit budget the transport's
    # sse_read_timeout uses (fit_budget.py), floored at _STEP_STALL_TIMEOUT_S for the non-fit steps
    # (a lead LLM call, an analyst sub-agent). A hung provider CALL is still caught fast by the
    # 120s client timeout; this watchdog only backstops a wedged step.
    from .fit_budget import fit_time_budget
    stall_timeout = max(_STEP_STALL_TIMEOUT_S, fit_time_budget(n_starts))
    # Self-imposed total-runtime budget (see the docstring): checked between super-steps, never
    # mid-step, so we always stop at a clean boundary rather than interrupting a live commit/fit.
    deadline = (time.monotonic() + max_wall_seconds) if max_wall_seconds is not None else None
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": "petab-mcp-run"}

    async def _early_stop_reason() -> str | None:
        """The server's early-stop reason if the run should stop now, else ``None`` (fail-open:
        any probe error just means "keep going")."""
        if progress_probe is None:
            return None
        try:
            res = await progress_probe("get_progress", {})
            sc = getattr(res, "structuredContent", None)
            if isinstance(sc, dict):
                data = sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
                if isinstance(data, dict) and data.get("should_stop"):
                    return data.get("stop_reason") or "early-stop condition met"
        except Exception:  # pragma: no cover - never let the probe break the run
            pass
        return None

    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    while True:
        retriable: str | None = None
        agen = None
        wall_budget_exceeded = False
        early_stopped: str | None = None
        try:
            # Open the stream INSIDE the try so a synchronous raise from astream (e.g. on the
            # input=None resume path) is handled/degraded here, not left to abort the whole run.
            agen = agent.astream(stream_input, config=config, stream_mode="values").__aiter__()
            while True:
                try:
                    await asyncio.wait_for(agen.__anext__(), timeout=stall_timeout)
                except StopAsyncIteration:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    wall_budget_exceeded = True
                    break
                # Deterministic early-stop: poll the server between super-steps and stop at this
                # clean boundary once the run has stopped improving (streak cap / plateau).
                early_stopped = await _early_stop_reason()
                if early_stopped is not None:
                    break
            stop_reason = (
                "wall_time_budget_exceeded" if wall_budget_exceeded
                else f"early_stop: {early_stopped}" if early_stopped
                else "completed"
            )
        except asyncio.TimeoutError:
            retriable = (f"stalled: no step completed in {stall_timeout:.0f}s "
                         "(hung provider call or wedged fit)")
        except _TRANSIENT_ERRORS as exc:  # call-level retries exhausted -> resume the run
            retriable = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"  # not retriable (recursion limit, etc.)
        finally:
            # Best-effort close; bound it so a truly wedged stream can't hang teardown too.
            if agen is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agen.aclose(), timeout=10)
        if wall_budget_exceeded:
            print(f"[agent] wall-time budget ({max_wall_seconds:.0f}s) exceeded — stopping "
                  "at this step boundary; the judge scores whatever is currently committed.")
            break
        if early_stopped is not None:
            print(f"[agent] early-stop: {early_stopped} — stopping at this step boundary; "
                  "the judge restores + scores the best-BIC checkpoint.")
            break
        if retriable is None:
            break
        if deadline is not None and time.monotonic() >= deadline:
            # Out of time even before a resume would start — don't burn a backoff on a retry
            # the deadline will just cut off anyway.
            stop_reason = f"wall_time_budget_exceeded (during retry: {retriable})"
            print(f"[agent] wall-time budget ({max_wall_seconds:.0f}s) exceeded while handling "
                  f"'{retriable}' — stopping instead of resuming.")
            break
        if resume_left <= 0:
            stop_reason = retriable  # out of resume attempts — end with whatever was built
            break
        resume_left -= 1
        backoff = min(5.0 * (_RUN_RESUME_ATTEMPTS - resume_left), 30.0)
        print(f"[agent] {retriable} — resuming from last checkpoint in {backoff:.0f}s "
              f"({resume_left} resume(s) left)")
        await asyncio.sleep(backoff)
        stream_input = None  # resume pending work from the checkpoint

    if breaker_state["n_tool_errors"]:
        disabled = sorted(breaker_state["disabled"])
        print(f"[agent] tool errors this run: {breaker_state['n_tool_errors']}"
              + (f"; disabled: {disabled}" if disabled else ""))
    return {
        "stop_reason": stop_reason,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
    }
