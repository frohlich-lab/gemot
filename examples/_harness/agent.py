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


_GOF_MIN_P = 0.05          # a fit with chi2 GoF p below this is statistically rejected
_REDUCED_CHI2_MAX = 2.0    # fallback gate when the GoF p-value is unavailable
_MAX_FINISH_REFUSALS = 2   # push back on finishing a rejected fit this many times, then relent


async def _current_gof(session) -> dict[str, Any]:
    """Fetch the CURRENT HEAD's goodness-of-fit from the server (ground truth, not the
    agent's self-report). Returns {} if unavailable so the gate fails open."""
    try:
        res = await session.call_tool("get_fit_summary", {})
        sc = getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            return sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
    except Exception:  # pragma: no cover - never let the gate break the run
        pass
    return {}


def _decision_tool(session):
    """A lead-only control tool that forces an explicit iterate / rollback / finish choice
    AND hard-gates ``finish`` on the server's real goodness-of-fit.

    The panel otherwise tends to make one edit and stop, and — the dominant sweep failure —
    to call ``finish`` while misreading the fit (declaring a chi2 GoF p of 1e-8 "noise").
    So ``finish`` is PUSHED BACK when the current HEAD is statistically rejected
    (chi2_gof_pvalue < 0.05, or reduced_chi2 ≫ 1 as a fallback): the decision is checked
    against the server's metrics, not the model's claim about them. The push-back is bounded
    — after ``_MAX_FINISH_REFUSALS`` rejected attempts the gate RELENTS and lets the run
    finish (recording that it never reached the noise floor), so a genuinely-stuck run ends
    cleanly instead of looping to the recursion limit. This kills the premature/misread
    finishes without trapping a run that truly cannot improve.
    """
    from langchain_core.tools import StructuredTool

    _guidance = {
        "iterate": "Recorded: iterate. Make the next single justified edit, then re-assess "
                   "the CURRENT HEAD's metrics.",
        "rollback": "Recorded: rollback. Find the BEST-scoring prior version with list_versions "
                    "(lowest bic / is_new_best, not just the previous one), read it with "
                    "get_version, and re-author it with set_model — then continue from there.",
        "finish": "Recorded: finish — the fit is statistically adequate (chi2_gof_pvalue >= "
                  "0.05, reduced_chi2 ≈ 1). Report the final neg_log_likelihood, bic, and "
                  "exactly what you changed, then stop.",
    }
    state = {"finish_refusals": 0}  # bounded push-back on finishing a rejected fit

    async def make_decision(action: str, rationale: str = "") -> str:
        act = (action or "").strip().lower()
        print(f"[lead decision] {act or '?'} — {rationale}")
        if act == "finish":
            gof = await _current_gof(session)
            p = gof.get("chi2_gof_pvalue")
            rchi2 = gof.get("reduced_chi2")
            # Fail CLOSED: if adequacy can't be verified (no GoF metrics — the fit summary
            # errored or a fit hasn't run), do not silently allow finish; treat it as
            # not-yet-adequate and push back (still bounded by _MAX_FINISH_REFUSALS below).
            if p is not None:
                rejected = p < _GOF_MIN_P
                detail = f"chi2_gof_pvalue={p:.3g} < {_GOF_MIN_P}"
            elif rchi2 is not None:
                rejected = rchi2 > _REDUCED_CHI2_MAX
                detail = f"reduced_chi2={rchi2:.3g} ≫ 1"
            else:
                rejected = True
                detail = "goodness-of-fit could not be verified (no fit metrics available)"
            if rejected:
                if state["finish_refusals"] < _MAX_FINISH_REFUSALS:
                    state["finish_refusals"] += 1
                    print(f"[finish BLOCKED {state['finish_refusals']}/{_MAX_FINISH_REFUSALS}] "
                          f"HEAD statistically rejected ({detail})")
                    return (
                        f"FINISH REFUSED ({state['finish_refusals']}/{_MAX_FINISH_REFUSALS}) — "
                        f"the current HEAD is statistically REJECTED ({detail}): its weighted "
                        "residuals are far larger than the known measurement noise, so the model "
                        "is still MIS-SPECIFIED regardless of how much nll improved. A tiny "
                        "chi2_gof_pvalue is NOT 'noise'. Keep iterating on the rate-law STRUCTURE "
                        "(or re-author the best prior version via get_version + set_model) to try "
                        "to reach chi2_gof_pvalue >= 0.05. "
                        "If, after genuinely trying, the fit truly cannot improve, you may finish."
                    )
                # exhausted the push-back: relent so the run ends cleanly, not at the step limit
                print(f"[finish ALLOWED] after {state['finish_refusals']} refusals; fit still "
                      f"rejected ({detail})")
                return (
                    f"Recorded: finish — NOTE the fit is STILL statistically rejected ({detail}) "
                    f"after {state['finish_refusals']} further attempts. Report the final "
                    "neg_log_likelihood/bic, state plainly that the model could NOT be brought to "
                    "the noise floor and what you changed, then stop."
                )
        return _guidance.get(act, "Unknown action — use one of: iterate, rollback, finish.")

    return StructuredTool.from_function(
        coroutine=make_decision,
        name="make_decision",
        description=(
            "Record your control decision after reading the CURRENT HEAD's fit metrics "
            "(bic / reduced_chi2 / chi2_gof_pvalue / is_new_best — never an earlier version's). "
            "`action`: 'iterate' (fit still inadequate — reduced_chi2 ≫ 1 or chi2_gof_pvalue < "
            "0.05 — re-author an improved model), 'rollback' (last change didn't help — re-author "
            "the best-bic prior version via get_version + set_model), or 'finish' (statistically "
            "adequate — chi2_gof_pvalue >= 0.05 AND "
            "reduced_chi2 ≈ 1). NOTE: 'finish' is programmatically REFUSED while the fit is "
            "rejected, so do not try to finish a poor fit. Prefer the FEWEST free parameters that "
            "pass (lower bic wins). Give a one-line `rationale`; call once per round after each edit."
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
    # union of the lead's + analysts' tools. This pool is only used to resolve each analyst's
    # tool objects; it is NOT what the lead gets (see lead_tools below).
    all_tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
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
        kwargs["middleware"] = [_builtin_tool_filter(), _history_trimmer()]
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
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": "petab-mcp-run"}
    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    while True:
        retriable: str | None = None
        agen = None
        try:
            # Open the stream INSIDE the try so a synchronous raise from astream (e.g. on the
            # input=None resume path) is handled/degraded here, not left to abort the whole run.
            agen = agent.astream(stream_input, config=config, stream_mode="values").__aiter__()
            while True:
                try:
                    await asyncio.wait_for(agen.__anext__(), timeout=stall_timeout)
                except StopAsyncIteration:
                    break
            stop_reason = "completed"
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
        if retriable is None:
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

    return {"stop_reason": stop_reason}
