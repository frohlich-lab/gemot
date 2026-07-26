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
import re
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
    # NOTE: get_fit_summary is deliberately NOT here. `fit` (and rollback/restore) already return
    # the whole fit block — metrics, bic, parameters_at_bound, the noise-sigma verdict + advice,
    # and the residual_structure verdict — so for the LEAD, which is the only agent that fits, a
    # summary call reads back what it was just handed. Trace analysis of sweep 50751606 measured 9
    # such calls against 451 fits: near-zero benefit for a tool schema re-sent every turn. The tool
    # stays on the server: the analysts (who never fit) and the harness's own finish gate and
    # final scoring all read the score through it. `describe_problem` covers has_unfit_edits.
    "get_fit_diagnosis",    # per-observable bias/chi2 + worst-N points — bounded residual read
    "get_identifiability",  # FIM SE/correlations + multistart spread — WHICH params to cut/merge
    "list_versions",        # scored version history — pick the BEST version to go back to
    "get_version",          # read a prior version's Antimony + params (to inspect it)
    # go BACK to an earlier model the cheap way: restore its stored fit instead of re-authoring
    # it (a re-author recompiles/refits to the same result and appends a duplicate — `fit` now
    # warns on, then refuses, a structure it has already seen). This is the fix for the
    # dominant sweep waste of re-encoding already-visited structures (see issue #59).
    "restore_version",      # restore a prior version's model + fit as the working state (no refit)
    "get_errors",
    # author the whole model in one step. `set_model` is the SINGLE model editor (issue #82): every
    # structural change is a whole-model re-author — the incremental editors (add_reaction /
    # change_rate_law / remove_reaction) were removed. `restore_version` (above) IS included:
    # "going back" via re-authoring the old model wastes a recompile+refit and appends a duplicate
    # (the top sweep waste — issue #59), so the lead is given the cheap restore instead.
    # It also carries the OPTIONAL bundle (issue #114): `parameters=[...]` applies the bound/scale
    # edits atomically with the re-author and `fit=true` scores them, so the dominant
    # set_model -> change_parameters -> fit cycle (55% of the lead's calls) costs ONE step.
    "set_model",            # build the first model AND re-author it wholesale in one step
    # score the assembled model. Edits are fit-free by default (issue #81): set_model /
    # change_parameters mutate the working model but do NOT fit unless asked — `fit` is the single
    # tool that fits + snapshots a version, and is still what the lead calls when a coherent change
    # spans SEVERAL edits (assemble, then fit once).
    "fit",
    # change parameters' ESTIMATION in place — the prompt tells the lead to WIDEN a binding
    # bound (or switch scale) rather than re-typing a value the multistart fit ignores. ONE batched
    # editor: bundle every bound/scale change of a coherent retuning into a single call (with
    # fit=true to score it), or pass them to set_model(parameters=[...]) when they belong with a
    # re-author.
    "change_parameters",
    # ...and READ them, so the lead can see what it is changing: id, bounds, scale, nominalValue,
    # plus each row's `role` (kinetic / noise_sigma / observable_scaling / initial_condition /
    # exponent / constant) and `status` (estimated / fixed). The prompt directs the lead here to
    # check whether a sigma_* is ESTIMATED, and it is the read that keeps a widening or a re-author
    # honest: a symbol introduced is added ESTIMATED and counts toward k however tight its bounds,
    # and an observation-model factor must never be wired into a rate law. One row per parameter,
    # so it grows with the model — the only lead tool not strictly size-bounded, kept because
    # change_parameters without it is an edit made blind.
    "get_parameters",
    # NOTE: the UNBOUNDED data/diagnostic tools (get_measurements, query_data, get_fitting_log,
    # list_data_tables) are deliberately NOT here — they belong to the analyst subagents, since
    # their large results would pile up in the lead's re-sent history (the input-token cost). The
    # lead instead has get_fit_diagnosis (a COMPACT, size-bounded residual read) for cheap
    # self-diagnosis, and DELEGATES the deep dives — full residual SQL, the optimizer log, raw
    # data characterization — to the analysts via `task`.
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
                  "exists (model_initialized: false) AND re-authors/replaces an existing one. Use "
                  "for the initial build and for any coupled/symmetric or multi-reaction STRUCTURAL "
                  "change in one coherent step; reconciles + lints, reverts on failure. PREFER THE "
                  "ONE-CALL FORM: parameters=[{parameter_id, lower_bound?, upper_bound?, scale?}, "
                  "...] applies those bound/scale edits atomically with the re-author (they may "
                  "target a parameter it introduces), and fit=true "
                  "scores them as ONE version — returned nested under `fit`, with "
                  "version/committed mirrored on top. Without fit= it is a fit-free EDIT. If the "
                  "edit applies but nothing is scored (applied: true, committed: false) the edit "
                  "STAYS in place — `fit.error` = a failed fit (repair it, don't re-apply); "
                  "`fit.no_change` = this structure already IS a version (change something else). "
                  "Warns if a "
                  "previously-estimated parameter is dropped. To go BACK to a model use "
                  "restore_version(n), not set_model. Args: antimony, justification, references?, "
                  "parameters?, fit?, n_starts?."),
    "change_parameters": ("Batch-edit the estimation BOUNDS and/or SCALE of parameters in ONE "
                          "atomic call — the search space, never a parameter's value. changes is a "
                          "non-empty list of {parameter_id, lower_bound?, upper_bound?, scale?} "
                          "entries (a bound edit gives BOTH bounds; scale is 'lin'|'log'|'log10'); "
                          "one rejected entry reverts the whole batch. Takes fit=/n_starts= like "
                          "set_model; pass these entries as set_model's parameters= when they "
                          "belong with a re-author. REJECTS widening an estimated noise sigma past "
                          "its data-driven cap, and a 3rd widening of a still-pinned parameter. "
                          "Args: changes, justification, references?, fit?, n_starts?."),
    "fit": ("Fit the current working model and snapshot a version. Needed "
            "on its own only when a coherent change spans SEVERAL edits: without fit=true they "
            "leave the model UNFIT (describe_problem -> has_unfit_edits: true), so make the edits, "
            "then call fit ONCE to score them together — a one-edit change is cheaper as "
            "set_model(..., fit=true), which runs this same fit. "
            "Fits the free parameters (pyPESTO multistart), simulates, and returns the fit block "
            "(neg_log_likelihood, chi2, bic, improvement, is_new_best) + a progress block + "
            "residual_structure (flagged_observables + summary: whether the model's FORM is still "
            "incomplete — read it HERE, do not spend a get_fit_diagnosis call to ask) + the "
            "noise-sigma verdict (noise_sigma_report/_inflated/_collapsed + advice) + the new "
            "version. Skips the refit and returns stored metrics if the model is identical to the "
            "current version (no_change: true). Args: n_starts? (REPLACES the server's default "
            "multistart count — describe_problem's fit_n_starts_default — so omit it unless you "
            "want MORE than that; below it fits LESS hard, 0 only simulates. The result's n_starts "
            "is what actually ran), justification?."),
    "describe_problem": "Orientation: whether a model exists (model_initialized), table/entity counts, current version, whether the working model has unfit edits (has_unfit_edits), and the multistart count a fit uses by default (fit_n_starts_default). Cheap.",
    "get_model_structure": "Structured model view: species, parameters, and reactions (id, reversibility, rate law) — empty until a model is built.",
    "get_parameters": ("Parameter table, one row per parameter: id, bounds, scale, nominalValue "
                       "(the fitted value), estimate flag, plus `role` (kinetic / noise_sigma / "
                       "observable_scaling / initial_condition / exponent / constant) and `status` "
                       "(estimated / fixed). Read `role` to keep roles straight: a noise_sigma or "
                       "observable_scaling parameter belongs to the OBSERVATION model, not the "
                       "kinetics, and must never appear in a rate law — whether or not it is "
                       "estimated. Read `status` before adding parameters: every ESTIMATED one "
                       "counts toward k and inflates bic however tight its bounds, and a symbol "
                       "you introduce is added estimated. Do NOT assume sigma_* or *_init are "
                       "fixed — noise sigmas are often estimated, and initial conditions are "
                       "estimated by DEFAULT (only an IC of exactly 0 or 1 stays fixed)."),
    "get_fit_summary": "Current fit metrics only (neg_log_likelihood, chi2, n_data_points) — a cheap score check.",
    "get_fit_diagnosis": ("The DETAIL behind the residual_structure verdict fit already gave you: "
                          "per-observable weighted chi2 + mean_weighted_residual (bias) + sign/runs "
                          "test p-values + worst-N points (measurement vs simulation). Call it to "
                          "see WHERE/HOW the fit is off — not to ask WHETHER it is (fit answers "
                          "that). Arg: worst_n. For a full custom breakdown or the optimizer log, "
                          "delegate to an analyst."),
    "get_identifiability": ("COMPACT identifiability diagnosis (bounded): per-parameter standard "
                            "error + fold_uncertainty, marginal & partial correlations, near-flat "
                            "FIM eigen-directions (non-identifiable parameter COMBINATIONS), and "
                            "multistart spread — with a ranked `candidates` list. Tells you WHICH "
                            "estimated parameters are superfluous when the model is too complex "
                            "(high bic / many params): fix, drop, or MERGE a collinear pair. Use it "
                            "BEFORE adding parameters and to prune toward parsimony. Arg: top_n."),
    "list_versions": ("List every model version (each a fit snapshot) with its kind, label, and fit "
                      "metrics (neg_log_likelihood, bic, is_new_best) — use it to find the "
                      "best-scoring version, then restore_version(v) to go back to it."),
    "get_version": ("Read a prior version's Antimony model + parameter table (to inspect it). To "
                    "GO BACK to an earlier model, call restore_version(v) — do NOT re-author it "
                    "with set_model. Arg: version."),
    "restore_version": ("Go BACK to an earlier model: restores version v's model + its stored fit "
                        "as the working state WITHOUT recompiling or refitting, and appends a new "
                        "version. Use this instead of re-authoring an earlier model with set_model "
                        "(that refits to the same result and makes a duplicate). Pick the target "
                        "with list_versions (lowest bic). Arg: version."),
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


# ---------------------------------------------------------------- biocurator (optional)
# A FOURTH analyst, added for the deepagents driver whenever a bio-knowledge server is reachable. Unlike
# the three above — which reason over THIS problem's model/data through the petab-mcp tools —
# the biocurator reaches OUTSIDE the problem, into curated biomedical knowledge bases served by
# a SEPARATE bio-knowledge MCP server (e.g. BioContextAI: STRING/KEGG/Reactome/OpenTargets +
# literature). Its tools are runtime-discovered from that server (their names are unknown ahead
# of time), so it is built dynamically in run_deepagents rather than listed in ANALYSTS.
#
# What it reports: what the named entities are, how they interact, and — crucially — which
# UNMEASURED (latent) intermediates, regulators, and processes are known to connect or act on
# them (a complex between two observed proteins, a phosphatase, a feedback loop, degradation,
# translocation). Those latent pieces are exactly what data-only reasoning misses, so surfacing
# them — as evidence-grounded CANDIDATES the lead still has to justify on bic — is a core reason
# to consult it.
#
# It is deliberately BLINDED to the modeling task. The benchmark examples ship the ORIGINAL
# published data and withhold only the model (examples/BENCHMARK_NOTES.md), so an analyst that
# knew it was reconstructing a model could look up the source paper and leak the answer. The
# blind is enforced two ways: (1) STRUCTURALLY — the biocurator holds ONLY the bio tools, never
# a petab tool that would reveal the data/task; (2) by PERSONA — it is framed purely as a
# general biology reference desk (entities, their interactions, and the latent entities/
# processes around them). Its prompt never mentions a model, data, fitting, or an experiment —
# not even to forbid them — so there is no frame for a stray hint in the lead's query to latch
# onto. The lead is told to pass it only neutral entity/observable NAMES (see
# scenario.DEFAULT_SYSTEM_PROMPT). What it returns is the domain knowledge a modeller brings, not the
# withheld mechanism.
_BIOCURATOR_REPORT_CONTRACT = (
    "\n\nIf you have lookup tools, ALWAYS call at least one before you conclude rather than "
    "answering from the request alone; if you have none, answer from your own knowledge and flag "
    "it as such. Then return a SHORT report: (1) what each named entity "
    "is; (2) the known interactions AMONG the named entities — direction and type of each "
    "(activates, inhibits, binds, phosphorylates, is a substrate of, …), with a PMID/DOI where "
    "one grounds the claim; (3) other entities and processes known to connect or act on them "
    "that are NOT in the list you were given — intermediates between them, and regulators or "
    "fluxes (an upstream activator, a feedback inhibitor, a phosphatase, production, "
    "degradation, transport between compartments) — each flagged as not in the given list; and "
    "(4) which claims are well-established vs speculative. Flag if an entity is unknown to you "
    "or the evidence is thin. NEVER return an empty message. Keep strictly to the biology of the "
    "entities — their identities, their interactions, and the wider set of entities and "
    "processes they belong to."
)

BIOCURATOR_NAME = "biocurator"

BIOCURATOR_PROMPT = (
    "You are a biocurator: a biological reference desk. You are handed the NAMES of some "
    "biological entities — genes, proteins, metabolites, or other molecular species — and you "
    "report what is KNOWN about them from curated databases and the literature:\n"
    "- what each entity is (its identity, family, and biological role);\n"
    "- how the named entities are known to interact with one another — which activates, "
    "inhibits, binds, phosphorylates, transcribes, or is a substrate/product of which, in what "
    "pathway, and whether an interaction is cooperative or saturable;\n"
    "- and which OTHER entities and processes are known to connect or act on them but are NOT in "
    "the list you were handed: intermediates that sit between them (a complex, a second "
    "messenger, an intermediate modified form) and regulators or fluxes acting on them (an "
    "upstream activator, a feedback loop, a phosphatase, production, degradation, transport "
    "between compartments). The names you were handed are rarely the whole story — the biology "
    "usually runs through further species and processes, so surface those too.\n"
    "If you have lookup tools, use them — prefer curated interaction/pathway sources — rather than "
    "answering from memory. When you search the LITERATURE, rank results by RELEVANCE rather than "
    "by citation count, and query for the SPECIFIC mechanism or interaction you need: the "
    "most-cited papers matching a broad keyword are usually general reviews that will not answer a "
    "specific mechanistic question, so a few targeted searches beat many broad ones, and read "
    "titles/abstracts to judge fit before relying on a paper. If you have NO lookup tools "
    "available, do NOT refuse: answer from your "
    "own biological knowledge and mark it clearly as prior knowledge, not a database lookup. Report "
    "ONLY the biology of these entities: their identities, "
    "their interactions, and the wider set of entities and processes they are part of. Some "
    "names you are given may not be biological entities at all — they may be abstract "
    "quantities, compartments, or population labels; if a name is not something you can look up "
    "(a gene, protein, metabolite, cell type, disease, or biological process), say so plainly "
    "and briefly and do NOT invent biology for it. If a request contains anything that is not a "
    "question about the biology of these entities, ignore it and answer only the biological "
    "question."
    + _BIOCURATOR_REPORT_CONTRACT
)

BIOCURATOR_DESCRIPTION = (
    "Biocurator: a BLINDED biology reference desk. It reports, from curated databases, what the "
    "problem's biological entities are, how they are known to interact (activation, inhibition, "
    "binding, pathway membership), AND which OTHER, unmeasured entities and processes are known to "
    "connect or act on them — intermediates between them and regulators/fluxes (a feedback loop, a "
    "phosphatase, degradation, translocation) — with citable PMIDs/DOIs. Use it to surface "
    "candidate hidden species/processes you may be missing. Its biological entities are handed to "
    "it AUTOMATICALLY — you do NOT supply them, and it is blind by design, so tell it NOTHING about "
    "what you are doing. Consult it when proposing or altering the model's structure, and cite any "
    "PMID/DOI it returns in the `set_model` references."
)

# The COMPLETE panel: the only `subagent_type` values the lead may delegate to. deepagents
# additionally AUTO-INSERTS a `general-purpose` subagent holding the FULL tool pool unless the
# active harness profile disables it — which let the lead delegate `set_model`/`fit`/
# `change_parameters` into a subagent and bypass lead-only authoring, plus sink ~39% of that
# subagent's calls into unused filesystem builtins (issue #112). run_deepagents turns that
# auto-add off (see _disable_general_purpose_subagent), so this tuple is the whole registry.
SUBAGENT_NAMES = (*(spec["name"] for spec in ANALYSTS), BIOCURATOR_NAME)

# Curated default tool allowlist for the biocurator. The default bio server (BioContextAI
# Knowledgebase) exposes ~50 tools across ~20 databases — literature search, clinical trials,
# drug labels, grants, cell-type markers, structures, … — and every tool's JSON schema is re-sent
# to the model on EVERY turn (the per-turn input cost the petab analysts are curated against). A
# numeric cap would be blunt (it can drop the very tool needed), so instead we hand the biocurator
# a small, DELIBERATE set covering its remit: entity identity + protein domains + interactions +
# pathways + GO biological processes (for latent processes/regulation). Deliberately excluded:
# free-text literature (europepmc/biorxiv/scholar — also the leak vector), clinical/drug/grants,
# AlphaFold/PRIDE/PanglaoDB/AntibodyRegistry. These are the exact tool names the BioContextAI
# server registers (v0.2.x); `--bio-tools` overrides this set, and `--bio-tools all` exposes
# everything. If NONE of these match the connected server (a different bio server), run_deepagents
# falls back to exposing all discovered tools with a warning (see below).
DEFAULT_BIO_TOOLS = (
    # identity / resolution
    "get_uniprot_id_by_protein_symbol",
    "get_uniprot_protein_info",
    "get_ensembl_id_from_gene_symbol",
    # protein domains / families
    "get_protein_domains",
    # protein–protein interactions
    "get_string_id",
    "get_string_interactions",
    # pathways (Reactome only; KEGG dropped — its BioContextAI tools fail on the hosted endpoint,
    # consistent with KEGG's academic-only / restricted programmatic access. Add the KEGG tools
    # back via --bio-tools if you point at a server that has valid KEGG access.)
    "get_reactome_info_by_identifier",
    # function / biological processes (latent processes & regulation)
    "get_go_terms_by_gene",
    # broader (non-gene/protein) biology so the curator also serves CELL-level immunology and
    # metabolic models — cell types, diseases/phenotypes, and metabolites — via ontology lookups.
    # (Purely compartmental dynamics — SIR/SEIR epidemiology, predator–prey ecology — have NO
    # biological database to draw on; there the curator correctly abstains, see BIOCURATOR_PROMPT.)
    "search_ontology_terms",
    "get_cell_ontology_terms",
    "get_chebi_terms_by_chemical",
    # cross-resource signaling/regulatory PRIOR KNOWLEDGE (OmniPath): directed, signed,
    # literature-curated interactions and causal links — a strong source for "which species
    # regulate which, and the latent regulators between them". Exposed when the OmniPath MCP
    # (explore.omnipathdb.org, in BIO_MCP_URLS) is reachable; this is its single read-only SQL tool.
    # Harmless (just doesn't match) if that server is unreachable. See saezlab/omnipath-next.
    "execute_sql_query_on_omnipath_db",
)

# Literature tools (EuropePMC + bioRxiv), added to the biocurator's set ONLY when the operator
# opts in with --bio-literature. Kept OUT of the default because free-text literature search over
# a benchmark problem's distinctive entity panel is the one path that could surface the withheld
# source paper (see examples/BENCHMARK_NOTES.md); the biocurator's default structured tools
# (interactions/pathways/ontologies/OmniPath) are general biology and don't carry that risk. On a
# NOVEL problem (the product path) literature grounding is unambiguously useful, hence the toggle.
LITERATURE_BIO_TOOLS = (
    "get_europepmc_articles",
    "get_europepmc_fulltext",
    "get_biorxiv_preprint_details",
    "get_recent_biorxiv_preprints",
)

# Per-bio-server tool-discovery (tools/list) timeout. A slow/hung bio server (e.g. a slow OmniPath
# tools/list) must not stall — or, once its session may already be dead, ABORT — run startup: on
# timeout or error the server's tools are skipped and the biocurator runs with whatever resolved
# (issue #86). The petab tools/list is deliberately NOT bounded this way (a petab failure is fatal).
_BIO_DISCOVERY_TIMEOUT_S = 30.0


def _biocurator_entity_block(names: "list[str]") -> str:
    """A blind-preserving prompt suffix that hands the biocurator the entity NAMES directly.

    The biocurator holds no petab tool and cannot fetch its own input the way the analysts do
    (agent.py structural blinding), and the deepagents ``task`` tool delivers only a free-text
    ``description`` — a channel the lead used to break by stuffing the names into a silently
    DROPPED ``prompt=`` argument (issue #91). So the harness injects the names directly (the
    scenario's hand-curated biological entities — see Scenario.biocurator_entities). Kept STRICTLY
    to bare entity names with NO mention of data/model/measurement/fitting, so the task-blinding
    persona is preserved (see BIOCURATOR_PROMPT). Empty -> "" (prompt unchanged)."""
    if not names:
        return ""
    return (
        "\n\nThe biological entities to report on for this request are: "
        + ", ".join(str(n) for n in names)
        + ". Report on these entities; if the request names additional entities, include those "
        "too."
    )


def _build_biocurator_subagent(bio_tools: list, entity_names: "list[str] | None" = None) -> dict[str, Any]:
    """Build the biocurator subagent spec from the RUNTIME-DISCOVERED bio-server tools.

    Unlike the three static analysts (whose tool NAMES are hard-coded and resolved from the
    petab server via ALLOWED_TOOLS), the biocurator's tools are whatever the bio-knowledge MCP
    server exposes — unknown until introspected — so they are passed as tool OBJECTS directly,
    bypassing ALLOWED_TOOLS/_select_tools. Holding ONLY these bio tools (no petab tool) is the
    structural half of the task-blinding. See run_deepagents.

    ``entity_names`` (the hand-curated biological entities — see Scenario.biocurator_entities) is
    injected into the persona prompt so the biocurator is handed its input DIRECTLY rather than
    relying on the lead relaying it through the ``task`` ``description`` (issue #91). ``None``/
    empty leaves the prompt unchanged (no entities to look up — e.g. an abstract problem).
    """
    prompt_key = _subagent_prompt_key()
    return {
        "name": BIOCURATOR_NAME,
        "description": BIOCURATOR_DESCRIPTION,
        prompt_key: BIOCURATOR_PROMPT + _biocurator_entity_block(entity_names or []),
        "tools": list(bio_tools),
    }


# ---------------------------------------------------------------- literature access tracking
# When --bio-literature is on, the biocurator can search EuropePMC/bioRxiv. We record each such
# call (the query it ran + the DOIs/PMIDs it got back) into a run-scoped log, surfaced as
# ``biocurator_literature`` in result.json — a provenance/audit trail of what the biocurator
# actually READ (which also lets a sweep audit whether it pulled a benchmark problem's source
# paper). The full request/response is also in the Phoenix trace; this is the compact, analysable
# view. Only the LITERATURE tools are wrapped (they are the only ones present under --bio-literature).
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]]+")
_PMID_RE = re.compile(r"pmid['\"\s:=>]{1,4}(\d{4,9})", re.IGNORECASE)


def _content_to_text(content: Any) -> str:
    """Flatten an MCP tool result's content (a string, or a list of text blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


# EuropePMC's ``get_europepmc_articles`` returns ``resultType=core`` records — for each of ~10
# hits it ships the FULL bibliographic record (author/affiliation lists, MeSH headings, grant
# lists, chemical/keyword lists, cross-references, full-text URL lists, citation counts). That is
# ~50k+ tokens per search of metadata the biocurator never uses to build an ODE model; across ~10
# searches it drove the biocurator's context past 120k tokens/turn (a whole crauste run's analyst
# input was ~800k, vs ~150k without literature). We keep only the fields that carry modelling
# signal — title, abstract, DOI/PMID, journal, year, a capped author line — and drop the rest. The
# FULL response is still recorded (DOIs/PMIDs) for the provenance audit trail before trimming.
_EPMC_KEEP_FIELDS = ("id", "source", "pmid", "pmcid", "doi", "title", "authorString",
                     "pubYear", "abstractText")
_EPMC_AUTHOR_CAP = 200


def _trim_europepmc_search(text: str) -> "str | None":
    """Slim a ``get_europepmc_articles`` (resultType=core) JSON response to just the fields a
    modeller needs. Returns the trimmed JSON string, or ``None`` if the text isn't the expected
    shape (leave it untouched) — best-effort, never raises."""
    import json
    try:
        data = json.loads(text)
        results = data["resultList"]["result"]
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(results, list):
        return None
    slim = []
    for r in results:
        if not isinstance(r, dict):
            continue
        row = {k: r[k] for k in _EPMC_KEEP_FIELDS if k in r}
        journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
        if journal:
            row["journalTitle"] = journal
        # Publication type (review vs primary article) lives under pubTypeList.pubType, not at the
        # top level — extract it explicitly so the biocurator can weigh source type.
        pub_type = (r.get("pubTypeList") or {}).get("pubType")
        if pub_type:
            row["pubType"] = pub_type
        authors = row.get("authorString")
        if isinstance(authors, str) and len(authors) > _EPMC_AUTHOR_CAP:
            suffix = "…et al."
            cap = max(0, _EPMC_AUTHOR_CAP - len(suffix))
            row["authorString"] = authors[:cap].rstrip(",; ") + suffix
        slim.append(row)
    out = {"hitCount": data.get("hitCount"), "resultList": {"result": slim}}
    # Preserve the (tiny) paging cursors so a cursor-based follow-up search still works.
    for k in ("nextCursorMark", "nextPageUrl"):
        if data.get(k) is not None:
            out[k] = data[k]
    return json.dumps(out, ensure_ascii=False)


def _wrap_literature_recorder(tools: list, log: list) -> list:
    """Wrap each literature tool's coroutine to append {tool, query, dois, pmids} to ``log`` on
    every call — the provenance trail of what the biocurator read. Mutates the tool objects in
    place (like the circuit breaker) so the recorder travels with the tool into the biocurator
    subagent; layered on top of the breaker wrapper, so it sees the actual returned result.
    Best-effort — a recording failure never breaks the tool call."""
    from functools import wraps

    for tool in tools:
        name = getattr(tool, "name", None)
        orig = getattr(tool, "coroutine", None)
        if name is None or orig is None:
            continue

        def _make(name, orig):
            @wraps(orig)
            async def _record(runtime=None, **arguments):
                result = await orig(runtime=runtime, **arguments)
                try:
                    content = result[0] if isinstance(result, tuple) else result
                    text = _content_to_text(content)
                    # Record the FULL response (complete DOI/PMID audit trail) BEFORE any trimming.
                    log.append({
                        "tool": name,
                        "query": arguments,
                        "dois": sorted(set(_DOI_RE.findall(text))),
                        "pmids": sorted(set(_PMID_RE.findall(text))),
                    })
                    # Hand the biocurator a slimmed article payload (drops resultType=core bloat
                    # the model never uses); the audit log above still has everything.
                    if name == "get_europepmc_articles":
                        slim = _trim_europepmc_search(text)
                        if slim is not None and len(slim) < len(text):
                            # Preserve the original content SHAPE (bare string vs list-of-blocks)
                            # so trimmed and untrimmed results are the same type downstream.
                            new_content = ([{"type": "text", "text": slim}]
                                           if isinstance(content, list) else slim)
                            return ((new_content,) + result[1:]
                                    if isinstance(result, tuple) else new_content)
                except Exception:  # pragma: no cover - provenance/trim must never break a tool call
                    pass
                return result

            try:  # keep the original signature so langchain still injects `runtime` correctly
                _record.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _record

        tool.coroutine = _make(name, orig)
    return tools


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


# A silent stop (the graph ends on a no-tool turn) is not an accepted way to finish: the driver
# bounces it back into the finish gate this many times before relenting so a genuinely stuck run
# still terminates (at the step/wall limit) rather than looping forever.
_MAX_SILENT_BOUNCES = 3

# After this many CONSECUTIVE finish attempts with no valid fit, the gate gives up: it flags the
# run (finish_state["gave_up"]) so the driver ends it promptly as no-model-built, rather than
# refusing forever while the model spams finish to the recursion limit. It is NOT an accepted
# finish — there is no fitted model to score — just a bounded, honest termination.
_MAX_NO_FIT_FINISH_ATTEMPTS = 5

# A finish on a model whose ESTIMATED noise sigma is implausible (inflated to absorb misfit, or
# collapsed to overfit the noise — the server flags this in get_fit_summary) is REFUSED with the
# fix-the-mechanism advice, so the agent doesn't submit a BIC that rests on a gamed noise model
# (the judge fails those anyway). Bounded like the no-fit gate: after this many consecutive refusals
# it relents (lets the finish through) so a model that genuinely can't reach a plausible sigma still
# terminates cleanly at its honest — losing — score, instead of spamming finish to the step limit.
_MAX_SIGMA_FINISH_REFUSALS = 3


def _structured(res) -> Any:
    """The tool's structured payload from a CallToolResult — unwrapping FastMCP's ``{"result": x}``
    envelope — or ``None`` when there is no structured dict. The one place the finish gate and the
    driver probes share for reading a tool result, so the unwrap rule lives in exactly one spot."""
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        return sc.get("result", sc) if set(sc.keys()) == {"result"} else sc
    return None


async def _current_fit(call_tool) -> dict[str, Any]:
    """Fetch the CURRENT HEAD's fit metrics from the server (ground truth, not the agent's
    self-report). ``call_tool`` is an ``await call_tool(name, args)`` handle — the finish gate
    passes the UN-TIMED probe so this internal poll doesn't inflate the tool clock / tool_calls
    metric. Returns {} if the metrics can't be read — the finish check then sees no valid fit and
    fails CLOSED: it refuses ``finish`` rather than letting the run end with no scored model."""
    try:
        data = _structured(await call_tool("get_fit_summary", {}))
        if isinstance(data, dict):
            return data
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


async def _current_diagnosis(call_tool) -> dict[str, Any]:
    """The current HEAD's residual diagnosis (``get_fit_diagnosis``), or {} if unreadable.
    ``call_tool`` is an ``await call_tool(name, args)`` handle (the finish gate passes the
    un-timed probe so this doesn't inflate the tool clock).

    Used to put the fit's residual-structure verdict in front of the lead at the finish
    decision, so a finish is a considered call on the diagnosis rather than a reflex."""
    try:
        data = _structured(await call_tool("get_fit_diagnosis", {}))
        if isinstance(data, dict):
            return data
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return {}


# Injected as a fresh user turn when a run ended cleanly having authored NO model (see
# _model_committed). It restates the task so the resumed graph builds something rather than
# repeating the silent stop.
_NO_MODEL_NUDGE = (
    "You ended the run without authoring a model — nothing was built, so there is nothing to "
    "score. The task is to BUILD a kinetic model of this system: call set_model with an Antimony "
    "model (reactions + rate laws for the observed species, plus any latent/unobserved driver "
    "state the data imply), then fit it. Do that now — do not stop until a fitted model exists."
)


# Injected when a run with a built model ended WITHOUT finishing through make_decision — a silent
# stop is not an accepted exit. Bare enforcement of the protocol (no coaching on what to fix): it
# points the model at the finish gate, which then requires a fitted model and shows the diagnosis
# before it will confirm.
_SILENT_FINISH_NUDGE = (
    "You ended the run without finishing it. A run is ONLY complete when you finish through "
    "make_decision('finish') — a silent stop is not accepted. Your model must have a CURRENT fit: "
    "if you have edited it since the last fit, call fit first. Then call make_decision('finish') "
    "— you will be shown the fit's diagnosis and asked to confirm. If the fit is inadequate, keep "
    "improving the model instead. Do not stop until you have finished through make_decision."
)


async def _model_committed(probe) -> bool:
    """Whether the run has authored a model (``describe_problem.model_initialized``).

    The driver uses this to catch a run that ended cleanly having built nothing: the lead
    emitted a no-tool turn and the deepagents graph terminated WITHOUT going through the finish
    gate (which only guards ``make_decision('finish')``) — so neither the finish gate nor the
    zero-tools relabel (a `task` delegation to the biocurator counts as a tool) sees it, and it
    would otherwise score as an empty k=0 "completed" run. Probes via the un-timed
    ``progress_probe`` so the check doesn't inflate the tool clock. Fails OPEN (returns True)
    whenever it cannot determine the state, so an unavailable/failing probe never nukes a run."""
    if probe is None:
        return True
    try:
        data = _structured(await probe("describe_problem", {}))
        # Only a payload that EXPLICITLY reports model_initialized decides the answer; a
        # malformed/partial payload (key absent) falls through to fail-open True, so an
        # unexpected response shape never triggers a spurious no-model resume.
        if isinstance(data, dict) and "model_initialized" in data:
            return bool(data["model_initialized"])
    except Exception:  # pragma: no cover - never let the check break the run
        pass
    return True


def _token_accountant():
    """A LangChain callback that sums token usage across EVERY LLM call in a run.

    Attached via the run config's ``callbacks``, which LangChain/LangGraph propagate into the
    analyst + biocurator subgraphs — so this totals the lead AND the subagents, matching
    Phoenix's per-trace rollup, but in-process so the counts can be written straight to
    result.json without re-deriving them from the (huge) span database. Records prompt,
    completion, and cache-read tokens; ``cache_read`` is the provider-reported cached-prompt
    count (``usage_metadata.input_token_details.cache_read``), which is what makes an offline
    cost estimate cache-aware. One instance is reused across resume iterations so the totals
    cover the whole run. Never raises — token accounting must not break a run.
    """
    from langchain_core.callbacks import AsyncCallbackHandler

    # AsyncCallbackHandler (not the sync base): the run is driven with astream, so LangChain
    # awaits the async on_llm_end INLINE on the event loop instead of dispatching a sync handler
    # to a threadpool. That serialises the counter updates — parallel subagent LLM completions
    # cannot interleave a non-atomic `+=` and lose an increment.
    def _as_int(x) -> int:
        """A token count coerced to a non-negative int, or 0 for anything unparseable — so one
        malformed field never aborts accounting for the rest of a response's records."""
        try:
            return max(0, int(x or 0))
        except (TypeError, ValueError):
            return 0

    class _Accountant(AsyncCallbackHandler):
        def __init__(self):
            self.prompt = self.completion = self.cache_read = self.n_calls = 0

        async def on_llm_end(self, response, **kwargs):  # noqa: D401 - LangChain hook
            try:
                found = False
                for gens in getattr(response, "generations", None) or []:
                    for gen in gens or []:
                        msg = getattr(gen, "message", None)
                        um = getattr(msg, "usage_metadata", None) if msg is not None else None
                        if not um:
                            continue
                        found = True
                        # Per-field coercion (not one int() over the whole record): a bad value
                        # in one field contributes 0 but does not discard the other fields, the
                        # other generations, or the llm_output fallback below.
                        self.prompt += _as_int(um.get("input_tokens"))
                        self.completion += _as_int(um.get("output_tokens"))
                        self.cache_read += _as_int(
                            (um.get("input_token_details") or {}).get("cache_read"))
                        self.n_calls += 1
                if not found:
                    # Older/non-usage_metadata responses carry an aggregate in llm_output.
                    tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
                    if tu:
                        self.prompt += _as_int(tu.get("prompt_tokens"))
                        self.completion += _as_int(tu.get("completion_tokens"))
                        self.cache_read += _as_int(
                            (tu.get("prompt_tokens_details") or {}).get("cached_tokens"))
                        self.n_calls += 1
            except Exception:  # pragma: no cover - accounting must never break a run
                pass

        def totals(self):
            return {
                "prompt_tokens": self.prompt,
                "completion_tokens": self.completion,
                "cache_read_tokens": self.cache_read,   # cached prompt tokens (for cache-aware cost)
                "total_tokens": self.prompt + self.completion,
                "n_llm_calls": self.n_calls,
            }

    return _Accountant()


def _decision_tool(session, finish_state=None, probe=None):
    """A lead-only control tool for the two non-default decisions: ``rollback`` and ``finish``.

    ``probe`` is the driver's UN-TIMED ``await probe(name, args)`` handle; the gate's internal
    get_fit_summary / get_fit_diagnosis polls go through it so they don't inflate the run's
    tool_calls / compute_seconds (falling back to the timed ``session`` handle when absent).

    Plain iteration needs no call — to keep going the lead simply makes the next ``set_model``
    edit — so the tool exists only to record a rollback or to gate a finish.

    ``finish`` is the ONLY legitimate way to end a run: a run that silently stops (a no-tool
    turn) is bounced back into this gate by the driver (see run_deepagents), so the gate cannot
    be sidestepped. To finish, the model MUST have a fitted model — a finish with no valid fit,
    or with edits made since the last fit, is REFUSED and never relented (the harness does not
    accept an unfitted model as finished; the run ends by other means — the step/wall limits —
    rather than by accepting an unscored result). A valid finish is a TWO-call handshake: the
    first call ARMS and presents the fit diagnosis (bic + residual structure), the second
    CONFIRMS — so a diagnosis is always shown before a run is allowed to end. ``finish_state``
    (a shared dict) records ``confirmed`` for the driver, so a properly-confirmed finish is not
    bounced.

    It deliberately does NOT gate on the χ² goodness-of-fit (chi2_gof_pvalue / reduced_chi2).
    Whenever the observable noise sigmas are estimated (as in the issue-#54 benchmarks — though
    not every problem frees them), χ² pins near the data-point count by construction and its
    p-value is uninformative, so gating ``finish`` on p ≥ 0.05 made the panel abandon its
    best-BIC model to chase an unreachable target (issue #54). Where χ² cannot be computed at
    all, the tool SAYS so rather than surfacing it as a fit rejection.
    """
    from langchain_core.tools import StructuredTool

    finish_state = finish_state if finish_state is not None else {}
    finish_state.setdefault("confirmed", False)
    finish_state.setdefault("gave_up", False)
    # Read HEAD's fit/diagnosis through the un-timed probe when the driver wires one, so the
    # gate's own polling doesn't count toward the tool clock / tool_calls metric; fall back to
    # the (timed) session handle otherwise.
    reader = probe if probe is not None else session.call_tool

    # `finish` is handled entirely in the branch below (every path there returns), so it needs
    # no entry here — only `rollback` and the unknown-action default ever reach _guidance.
    _guidance = {
        "rollback": "Recorded: rollback. Find the BEST-scoring prior version with list_versions "
                    "(LOWEST bic / is_new_best, not just the previous one) and bring it back with "
                    "restore_version(v) — do NOT re-author it with set_model (that refits to the "
                    "same result and makes a duplicate). Then continue from there.",
    }
    # `armed_token` identifies the fit a finish was ARMED on (the diagnosis has been shown and a
    # confirming second finish is awaited), or None when not armed. The token is the fit's
    # identity, so an edit + refit between the two calls re-arms (the shown diagnosis would
    # otherwise be stale). It is NOT the get_fit_summary "version": that is None after a
    # rollback/restore_version (the persisted per-version sim blob carries no version), and
    # `None != None` would then skip arming and confirm on the first call — bypassing the gate on
    # the common rollback→finish path. bic+nll identify the scored fit whether or not a version
    # is attached.
    state = {"armed_token": None, "no_fit_refusals": 0, "sigma_refusals": 0}

    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    async def make_decision(action: str, rationale: str = "") -> str:
        act = (action or "").strip().lower()
        print(f"[lead decision] {act or '?'} — {rationale}")
        if act == "finish":
            fit = await _current_fit(reader)
            nll = fit.get("neg_log_likelihood")
            bic = fit.get("bic")
            version = fit.get("version")
            # Require a finite BIC — the score the harness actually finishes on. A finite nll is
            # necessary but not sufficient: get_fit_summary reports bic=None when n_data_points/k
            # are incomplete, and finishing then would report "bic=n/a" while the refusal text
            # promises "there is no bic to report". Gate on the score that must exist.
            has_fit = _finite(nll) and _finite(bic)
            # Block finish when there is no valid fit to finish on (nothing built, a non-finite
            # objective, or no computable bic) — a genuine failure, not a goodness-of-fit
            # judgement. We do NOT gate on chi2_gof_pvalue/reduced_chi2 (see the docstring): they
            # are uninformative when the noise sigmas are estimated, and chasing p ≥ 0.05 costs
            # the best-BIC model.
            if not has_fit:
                # NEVER relented into an accepted finish: the harness does not record an unfitted
                # model as finished (there is no bic to score). But it does not refuse FOREVER
                # either — after _MAX_NO_FIT_FINISH_ATTEMPTS consecutive no-fit attempts it GIVES
                # UP: flags the run so the driver ends it promptly as no-model-built, instead of
                # letting the model spam finish to the recursion limit.
                state["armed_token"] = None
                state["no_fit_refusals"] += 1
                if state["no_fit_refusals"] >= _MAX_NO_FIT_FINISH_ATTEMPTS:
                    finish_state["gave_up"] = True
                    print(f"[finish GAVE UP] no fittable model after {state['no_fit_refusals']} "
                          "attempts — ending the run as no-model-built")
                    return (
                        f"No fittable model after {state['no_fit_refusals']} finish attempts — "
                        "this run has no scoreable result and is being ended as no-model-built. "
                        "Stop now."
                    )
                print(f"[finish BLOCKED {state['no_fit_refusals']}/{_MAX_NO_FIT_FINISH_ATTEMPTS}] "
                      "no valid fit (no model built, non-finite objective, or no bic)")
                return (
                    "FINISH REFUSED — there is no valid fit to finish on: no model has been built "
                    "yet, or the fit has no finite bic (a non-finite objective, or an incomplete "
                    "n/k). A run is NOT accepted as finished without a FITTED model. Build a model "
                    "that simulates and fits, then finish on its bic."
                )
            # A valid fit exists -> reset the no-fit give-up counter.
            state["no_fit_refusals"] = 0
            # A model edited since the last fit has STALE metrics: the bic above belongs to the
            # previous structure, not the working one, so finishing now would score the wrong
            # model. Refuse until it is refitted. Naturally escapable (call fit) and version-
            # tracked, so it cannot loop indefinitely.
            if fit.get("has_unfit_edits"):
                state["armed_token"] = None
                print("[finish BLOCKED] unfit edits since last fit — metrics are stale")
                return (
                    "FINISH REFUSED — the working model has edits made SINCE the last fit, so "
                    "the reported bic belongs to the previous structure, not this one. Call fit "
                    "to score the current model, then finish on its bic."
                )
            # HARD STOP on a gamed noise model: an inflated/collapsed ESTIMATED sigma makes the
            # good-looking bic rest on an implausible noise model (the judge fails those). Refuse the
            # finish with the fix-the-mechanism advice so the agent addresses the FORM, not the noise
            # — but bounded (like the no-fit gate) so a model that genuinely can't reach a plausible
            # sigma still terminates at its honest score instead of spamming finish. A plausible
            # sigma resets the counter, so a real fix immediately clears the block.
            sigma_inflated = bool(fit.get("noise_sigma_inflated"))
            sigma_collapsed = bool(fit.get("noise_sigma_collapsed"))
            if sigma_inflated or sigma_collapsed:
                if state["sigma_refusals"] < _MAX_SIGMA_FINISH_REFUSALS:
                    state["armed_token"] = None
                    state["sigma_refusals"] += 1
                    advice = fit.get("noise_sigma_advice") or (
                        "the estimated noise sigma is implausible.")
                    # Name BOTH tails when both are present (one observable inflated, another
                    # collapsed) so the agent doesn't fix one and get re-blocked on the other.
                    which = " and ".join(w for w, on in (("inflated", sigma_inflated),
                                                         ("collapsed", sigma_collapsed)) if on)
                    print(f"[finish BLOCKED {state['sigma_refusals']}/{_MAX_SIGMA_FINISH_REFUSALS}] "
                          f"noise sigma {which} — refusing finish on a gamed noise model")
                    return (
                        f"FINISH REFUSED — this fit's bic rests on an implausible noise model: "
                        f"{advice} Fix the model FORM and refit; a finish resting on {which} noise "
                        "is not accepted. (the fit result's noise_sigma_report, and "
                        "get_fit_diagnosis, flag which observable.)"
                    )
                print(f"[finish RELENTED] noise sigma still implausible after "
                      f"{state['sigma_refusals']} refusals — allowing finish at its honest score")
            else:
                state["sigma_refusals"] = 0
            # Valid, fitted model. First finish call ARMS: show the residual diagnosis and
            # require a confirming second finish. This forces the finish to be a considered call
            # on the fit's residual STRUCTURE (sign/runs tests — the signal that survives
            # estimated sigmas) rather than a reflex. The token identifies THIS fit (bic+nll,
            # plus version as a tiebreak) so an edit + refit — or a rollback, which leaves
            # version None but changes bic/nll — re-arms rather than confirming a stale diagnosis.
            token = (bic, nll, version)
            if state["armed_token"] != token:
                state["armed_token"] = token
                diag = await _current_diagnosis(reader)
                rs = diag.get("residual_structure") or {}
                struct = rs.get("summary") or "residual-structure tests unavailable."
                per = diag.get("per_observable") or []
                worst = max(per, key=lambda o: abs(o.get("mean_weighted_residual") or 0.0),
                            default=None) if per else None
                worst_line = ""
                if worst is not None:
                    st = worst.get("sign_test") or {}
                    rtst = worst.get("runs_test") or {}
                    worst_line = (f" Worst observable {worst.get('observable_id')}: "
                                  f"sign-test p={st.get('p')}, runs-test p={rtst.get('p')}.")
                print(f"[finish ARMED v={version}] bic={bic:.6g} — presented diagnosis, "
                      "awaiting confirming finish")
                return (
                    f"FINISH ARMED — review before confirming. Current fit: bic={bic:.6g}, "
                    f"neg_log_likelihood={nll:.6g}. Residual structure: {struct}{worst_line} "
                    "If systematic structure is flagged, the model's FORM is still missing "
                    "something — keep improving BIC instead of finishing. If you have genuinely "
                    "bottomed out BIC and are satisfied with the residuals, call finish AGAIN to "
                    "confirm and end the run."
                )
            # Second finish on the same fit -> confirmed. Record it so the driver does not bounce
            # a legitimately-finished run back into the gate.
            state["armed_token"] = None
            finish_state["confirmed"] = True
            chi2_note = ""
            if fit.get("chi2_gof_pvalue") is None and fit.get("reduced_chi2") is None:
                chi2_note = (" (χ² goodness-of-fit is not reported here — n - k <= 0 or chi2 is "
                             "unavailable, so it could not be computed; this is NOT a fit rejection.)")
            print(f"[finish CONFIRMED v={version}] bic={bic:.6g}")
            return (
                f"Recorded: finish — final bic={bic:.6g}, neg_log_likelihood={nll:.6g}."
                f"{chi2_note} Report the bic, the neg_log_likelihood, and exactly what you "
                "changed, then stop."
            )
        # Any non-finish decision (rollback) disarms a pending finish: the state has changed, so
        # a later finish must re-present the diagnosis rather than confirm against a stale one.
        state["armed_token"] = None
        return _guidance.get(act, "Unknown action — use one of: rollback, finish.")

    return StructuredTool.from_function(
        coroutine=make_decision,
        name="make_decision",
        description=(
            "Record a control decision after reading the CURRENT HEAD's fit metrics "
            "(bic / neg_log_likelihood / is_new_best — never an earlier version's). You do NOT "
            "call this to keep iterating — to iterate, just make the next set_model edit. Call it "
            "only for `action`: 'rollback' (the last change didn't help — go back to the LOWEST-bic "
            "prior version with restore_version) or 'finish' (residuals show no systematic "
            "structure AND bic has bottomed out). BIC is the ONLY success target; chi2_gof_pvalue "
            "/ reduced_chi2 are UNINFORMATIVE when the noise sigmas are estimated — do NOT keep "
            "iterating just to chase chi2_gof_pvalue ≥ 0.05; use the sign_test / runs_test in "
            "get_fit_diagnosis (they read residual SIGNS, so they still detect bias/time-trend "
            "when chi2 is flat). 'finish' takes TWO calls: the first ARMS it and shows you the "
            "current bic and residual-structure diagnosis; review it, then call 'finish' AGAIN to "
            "confirm and end. 'finish' is refused when there is no valid fit yet (no model built / "
            "no finite bic) OR the model has edits since its last fit (call fit first — the "
            "reported bic is stale otherwise). Prefer the FEWEST free parameters that fit "
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


# --- drop deepagents' auto-added `general-purpose` subagent -------------------------------
# On top of the subagents we pass, deepagents AUTO-INSERTS one named `general-purpose` holding
# the FULL tool pool (deepagents 0.6.12 graph.py: it is added unless the active HarnessProfile
# sets `general_purpose_subagent.enabled = False`, or the caller supplies a spec of that name).
# We never curated it, and it broke the panel two ways (issue #112, sweep 50751606):
#   * containment leak — the lead delegated to it and it called `set_model` (9x), `fit` and
#     `change_parameters` INSIDE a subagent, bypassing lead-only authoring;
#   * waste sink — 84 of its 215 tool calls (39%) were filesystem builtins (glob/ls/read_file/
#     grep), which have no role in a PEtab modeling task.
# This is a different mechanism from the tool-pool filter below: the SUBAGENT is removed, not
# its tools. We register a provider-level HarnessProfile because deepagents resolves the
# profile for a PRE-BUILT model (which is what we hand it) via the model's `ls_provider` —
# "openai" for ChatOpenAI, which every model in this harness runs through (OpenRouter is
# reached via ChatOpenAI's base_url). Registration is additive and merges with deepagents' own
# per-model built-in profiles, so it does not clobber them.
_gp_subagent_disabled = False


def _disable_general_purpose_subagent() -> bool:
    """Turn off deepagents' auto-added ``general-purpose`` subagent. Idempotent.

    Returns True when the profile was registered. Warns and returns False if the installed
    deepagents doesn't expose the harness-profile API (a version drift) — the run continues
    with the roster deepagents builds, and ``_general_purpose_enabled`` reports the outcome.
    """
    global _gp_subagent_disabled
    if _gp_subagent_disabled:
        return True
    try:
        from deepagents import (GeneralPurposeSubagentProfile, HarnessProfile,
                                register_harness_profile)
        register_harness_profile(
            "openai",
            HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
        )
    except Exception as exc:  # pragma: no cover - guards against a deepagents API drift
        print(f"[agent] WARNING: could not disable deepagents' `general-purpose` subagent "
              f"({type(exc).__name__}: {exc}); the lead may be able to delegate model authoring "
              "to it (issue #112).")
        return False
    _gp_subagent_disabled = True
    return True


def _general_purpose_enabled(llm) -> "bool | None":
    """Would deepagents still auto-add the ``general-purpose`` subagent for ``llm``?

    Resolved through the SAME lookup ``create_deep_agent`` uses, so this answers the question
    the profile registration was meant to settle rather than approximating it. ``None`` means
    the private resolver is gone (API drift) and the answer is unknown.
    """
    try:
        from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
        profile = _harness_profile_for_model(llm, None).general_purpose_subagent
    except Exception:  # pragma: no cover - guards against a deepagents API drift
        return None
    return not (profile is not None and profile.enabled is False)


# deepagents injects a set of built-in filesystem + todo tools (ls/glob/read_file/write_file/
# edit_file/grep/execute/write_todos) into every agent, on top of the `tools` we pass. This
# task has no filesystem — the real tools are the MCP tools — so those built-ins are pure
# overhead: their JSON schemas are re-sent on every turn (the input-token cost this module
# works hard to keep lean, see _trim_descriptions/TOOL_DESCRIPTIONS), and each call the model
# wastes on them also burns a recursion step. `task` (subagent delegation) is NEVER excluded —
# the panel depends on it. We attach this filter to the LEAD only: neither the per-model
# `HarnessProfile.excluded_tools` route nor a per-subagent `middleware` list reliably reaches
# deepagents 0.6.12's declarative analyst subagents (both were verified to still leak FS calls
# there), so the four curated analysts keep their built-ins for now — the lead is the heavy
# tool-caller and is the one worth trimming. Trimming the analysts cleanly needs an upstream
# deepagents fix. Note this caveat no longer covers the biggest FS-call sink: the uncurated
# `general-purpose` subagent is removed outright above, not filtered.
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


# Transport/connection failures — a hung or closed MCP server (in practice a flaky HOSTED bio
# server: BioContextAI / OmniPath). These are NOT retryable tool errors: the model can't fix an
# unreachable server, and letting one propagate aborts the whole run (9 runs across sweeps died as
# `stop_reason: ClosedResourceError`, several at n_fits=0). anyio stream errors are matched by class
# NAME (so we don't hard-import anyio internals that vary across versions); the httpx family is
# matched against its public TransportError base in _is_transport_error, so the breaker classifies
# the SAME httpx surface the resume backstop does rather than a hand-listed subset that can drift.
_TRANSPORT_ERROR_NAMES = frozenset({
    "ClosedResourceError", "BrokenResourceError", "EndOfStream",           # anyio streams
})


def _is_transport_error(exc: BaseException) -> bool:
    """True for a transport/connection failure (an unreachable/closed MCP server) as opposed to a
    normal tool error the model should see and retry. Covers the anyio stream errors (by name), the
    builtin ConnectionError, and the WHOLE httpx.TransportError family (ConnectError/ReadError/
    WriteError/ReadTimeout/ConnectTimeout/RemoteProtocolError/PoolTimeout/WriteTimeout/… ) — the same
    surface the run-loop resume backstop matches, so the two never disagree on what is 'transport'."""
    if type(exc).__name__ in _TRANSPORT_ERROR_NAMES or isinstance(exc, ConnectionError):
        return True
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx always present with mcp
        return False
    return isinstance(exc, httpx.TransportError)


def _new_breaker_state() -> dict[str, Any]:
    """Mutable, run-scoped state shared by the wrapped tools and the lead's tool filter."""
    return {"disabled": set(), "n_tool_errors": 0, "streaks": {}, "events": []}


def _breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """The terminal content block returned in place of a disabled tool's result."""
    return [{"type": "text", "text": (
        f"TOOL DISABLED: `{name}` failed repeatedly with the same error and is disabled for the "
        f"rest of this run. Do NOT call it again — {hint}"
    )}]


def _transport_breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """Degraded block for a tool disabled after a transport failure (an unreachable/closed server).

    Distinct from :func:`_breaker_message`: a transport error disables on the FIRST failure, so the
    "failed repeatedly with the same error" wording would misdescribe it."""
    return [{"type": "text", "text": (
        f"TOOL DISABLED: `{name}`'s server is unreachable or closed (transport error) and it is "
        f"disabled for the rest of this run. Do NOT call it again — {hint}"
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


def _wrap_tools_with_circuit_breaker(tools: list, state: dict[str, Any],
                                     transport_disable_names: set[str] | None = None) -> list:
    """Wrap each tool's coroutine so repeated identical failures disable it (see above).

    Mutates the tool objects in place (and returns the list) so the breaker rides along into
    the analyst subagents, which hold these same objects. Best-effort: a tool without a
    coroutine is left untouched.

    ``transport_disable_names`` is the set of tool names for which a SINGLE transport error
    (unreachable/closed server) disables the tool immediately, without waiting for the 3-strike
    breaker. This is scoped to the EXTERNAL, optional bio tools (flaky hosted servers the model
    can't fix); a transport error on a CORE petab tool (the local server) instead falls through to
    the normal 3-strike / re-raise path, so one localhost blip can't silently strip ``fit`` /
    ``set_model`` for the rest of the run.
    """
    import inspect
    from functools import wraps

    transport_disable = transport_disable_names or set()

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
                    tripped = _record_error(state, name, f"{type(exc).__name__}: {exc}")
                    # A transport error (unreachable/closed server) on an EXTERNAL bio tool can't be
                    # fixed by the model and aborts the whole run if it propagates — disable that
                    # tool for the rest of the run and return the degraded message NOW, without
                    # waiting for the 3-strike breaker. (Mid-run counterpart to the bio-session
                    # isolation that already handles connect/discovery failures.) A transport error
                    # on a CORE petab tool is NOT auto-disabled here: it falls through to the normal
                    # 3-strike / re-raise path, so a single localhost blip can't strip `fit` or
                    # `set_model` for the rest of the run.
                    if name in transport_disable and _is_transport_error(exc):
                        state["disabled"].add(name)
                        return (_transport_breaker_message(name, hint), None)
                    if tripped:
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

# NOTE: the lead used to run a ContextEditingMiddleware (ClearToolUsesEdit) that cleared old
# tool results once its context passed ~90k tokens — a guard against an early failure mode where
# re-sent tool RESULTS ballooned a run to ~1.1M tokens and hanging calls. It was removed: all
# target models have a 1M-token window and the recursion limit already caps a run at ~120k tokens
# (a full crauste sweep of 17 runs topped out at 118k and the trimmer fired on only 2 of them,
# once each). With prompt caching the cleared history was ~98% near-free cache_read, while the
# clear itself invalidated the cached prefix — so trimming was net token-negative. The per-request
# timeout still backstops any slow call. Re-add a trimmer only if a model with a smaller window
# is introduced.


# --- tool-dispatch guard: make sure proposed tool calls actually FIRE -----------------------
# A whole class of runs (originally seen on the blasi zero-shot build) never executed a single
# tool: the lead kept emitting text-only assistant turns — narrating what it *would* do — and the
# ReAct loop, seeing an assistant message with no tool_calls, treated each one as "done" and ended
# with nothing built (n_tool=0, terminating in a handful of steps, nowhere near the recursion
# limit). Two things harden the loop against that here (orthogonal to the repeated-error circuit
# breaker above, which handles tools that keep FAILING — this handles tools that never fire AT ALL):
#   1. de-duplicate identical tool calls WITHIN a batch: some providers emit the same (name, args)
#      twice — occasionally with a repeated id — in one assistant turn, which wastes a fit and can
#      break the tool-call/result pairing on the follow-up request. We collapse exact duplicates to
#      the first occurrence before they dispatch, so results feed back cleanly.
#   2. don't let an idle (no-tool) turn silently END the run while NOTHING has fired yet: push back
#      with a corrective message and loop back to the model, up to K times, so the model gets a real
#      chance to emit an actual tool call. If it still hasn't after K nudges we stop cleanly and the
#      run is labelled ``no_tools_executed`` (see run_deepagents) — a degenerate run that is
#      EXCLUDED as such rather than left as an ambiguous, silent early "completion".
# The nudge fires ONLY while zero tools have executed, so it can never interfere with a genuine
# finish (which always comes AFTER set_model + fits have run).
_MAX_IDLE_MODEL_STEPS = 3   # consecutive no-tool turns to push back on before giving up

_IDLE_NUDGE = (
    "Your last message proposed NO tool call, so nothing happened. You cannot make progress by "
    "reasoning in prose — you must ACT through the tools. Emit a tool call now: start by orienting "
    "with describe_problem / get_model_structure, delegate data characterisation with the `task` "
    "tool, or author the model with set_model. Do not reply with text alone."
)


def _tool_call_key(tc: dict) -> str:
    """A stable identity for a proposed tool call: its name + canonicalised arguments.

    Two calls with the same name and the same arguments are the same request regardless of the
    id the provider stamped on them, so they compare equal here.
    """
    import json

    args = tc.get("args")
    try:
        args_norm = json.dumps(args, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - non-serialisable args fall back to repr
        args_norm = repr(args)
    return f"{tc.get('name')}\x00{args_norm}"


def _dedupe_tool_calls(tool_calls: "list[dict] | None") -> "tuple[list[dict], int]":
    """Drop exact-duplicate (same name+args) tool calls, keeping the FIRST of each.

    Accepts ``None`` (treated as empty). Returns ``(kept, n_dropped)``. Pure — no langchain
    types — so it is unit-testable without the agent stack.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    for tc in tool_calls or []:
        key = _tool_call_key(tc)
        if key in seen:
            continue
        seen.add(key)
        kept.append(tc)
    return kept, len(tool_calls or []) - len(kept)


def _dedupe_raw_tool_calls(raw: "list[dict] | None") -> list[dict]:
    """De-dupe the OpenAI-format raw tool-call list from ``additional_kwargs``.

    Mirrors :func:`_dedupe_tool_calls` but on the provider's raw shape
    (``{"id", "type", "function": {"name", "arguments"}}``), keying on name + the raw arguments
    STRING and keeping the first of each — so it collapses a repeated call regardless of whether
    the provider reused its ``id`` (matching by id membership alone would keep both copies of a
    repeated-id duplicate).
    """
    seen: set[tuple] = set()
    kept: list[dict] = []
    for c in raw or []:
        fn = c.get("function") or {}
        key = (fn.get("name"), fn.get("arguments"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


def _tool_dispatch_guard(stats: dict[str, Any], *, max_idle_steps: int = _MAX_IDLE_MODEL_STEPS):
    """Lead-only middleware implementing the two hardening steps described above.

    ``stats`` is a caller-owned dict the middleware mutates so run_deepagents can inspect the
    outcome after the run: ``tools_executed`` (real tool dispatches, incl. ``task`` delegations),
    ``dropped_dup_tool_calls`` (batch duplicates collapsed), ``idle_nudges`` (no-tool turns pushed
    back on), and ``gave_up_no_tools`` (True once the push-back budget was exhausted). Defined
    lazily so importing this module doesn't require langchain — only the deepagents driver does.
    """
    from langchain.agents.middleware.types import AgentMiddleware, hook_config
    from langchain_core.messages import AIMessage, HumanMessage

    def _dedupe_response(response):
        # response is a ModelResponse (has .result: list[BaseMessage]) or a bare AIMessage.
        msgs = getattr(response, "result", None)
        target = msgs[-1] if msgs else (response if isinstance(response, AIMessage) else None)
        if isinstance(target, AIMessage) and getattr(target, "tool_calls", None):
            kept, dropped = _dedupe_tool_calls(target.tool_calls)
            if dropped:
                stats["dropped_dup_tool_calls"] += dropped
                # Keep additional_kwargs["tool_calls"] (the raw provider copy) consistent with the
                # trimmed .tool_calls — langchain-openai serialises from .tool_calls when it is
                # non-empty, but collapse the raw copy the same way so nothing can resurrect a
                # duplicate (de-duping by name+args, not id, since a repeated call may reuse its id).
                ak = dict(getattr(target, "additional_kwargs", None) or {})
                if isinstance(ak.get("tool_calls"), list):
                    ak["tool_calls"] = _dedupe_raw_tool_calls(ak["tool_calls"])
                new = target.model_copy(update={"tool_calls": kept, "additional_kwargs": ak})
                if msgs:
                    msgs[-1] = new
                else:
                    return new
        return response

    def _guard(state):
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
            return None  # a tool call was proposed (or not an AI turn) — let it dispatch/route
        if stats["tools_executed"] > 0:
            return None  # a no-tool turn AFTER tools have run is a genuine finish — allow it
        # Degenerate: nothing has fired yet and the model answered with text only.
        if stats["idle_nudges"] < max_idle_steps:
            stats["idle_nudges"] += 1
            print(f"[tool-dispatch guard] no tool has fired yet and the model replied with text "
                  f"only — pushing back ({stats['idle_nudges']}/{max_idle_steps})")
            return {"messages": [HumanMessage(content=_IDLE_NUDGE)], "jump_to": "model"}
        stats["gave_up_no_tools"] = True
        print("[tool-dispatch guard] the model never dispatched a tool after "
              f"{max_idle_steps} push-backs — ending the run (no_tools_executed)")
        return {"jump_to": "end"}

    class _ToolDispatchGuard(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return _dedupe_response(handler(request))

        async def awrap_model_call(self, request, handler):
            return _dedupe_response(await handler(request))

        def wrap_tool_call(self, request, handler):
            out = handler(request)
            stats["tools_executed"] += 1
            return out

        async def awrap_tool_call(self, request, handler):
            out = await handler(request)
            stats["tools_executed"] += 1
            return out

        @hook_config(can_jump_to=["model", "end"])
        def after_model(self, state, runtime):
            return _guard(state)

        @hook_config(can_jump_to=["model", "end"])
        async def aafter_model(self, state, runtime):
            return _guard(state)

    return _ToolDispatchGuard()


# --- task-arg guard: reject a `task` call carrying kwargs the tool silently drops --------------
# The deepagents `task` tool takes ONLY (description, subagent_type). The lead used to assemble the
# biocurator's entity names into a `prompt=` argument, which `task` SILENTLY DISCARDS — so the
# biocurator (which, unlike the analysts, holds no petab tool and cannot self-serve) received only
# the short `description` and its whole input channel was broken (issue #91). The entity names are
# now injected into the biocurator directly (hand-curated per scenario; _build_biocurator_subagent);
# this guard closes the other half: it makes the dropped-kwarg bug LOUD instead of silent by
# bouncing such a call back to the model with a corrective error, so the model re-issues it with
# everything in `description` (this protects the analysts' `prompt=` drops too).
_TASK_ALLOWED_ARGS = frozenset({"description", "subagent_type"})
_MAX_TASK_ARG_REJECTIONS = 3   # bounded push-back on unexpected task kwargs per run, then relent


def _unexpected_task_kwargs(args: "dict | None") -> "list[str]":
    """The keys a ``task(...)`` call carried BEYOND (description, subagent_type) — e.g. a ``prompt=``
    the deepagents task tool silently drops. Order-stable. Pure — unit-testable without langchain."""
    if not isinstance(args, dict):
        return []
    return [k for k in args if k not in _TASK_ALLOWED_ARGS]


def _task_arg_reject_message(bad_keys: "list[str]") -> str:
    """The corrective error returned to the model in place of dispatching a mis-called ``task``."""
    return (
        "TASK CALL REJECTED — `task` accepts ONLY `description` and `subagent_type`; it received "
        f"unexpected argument(s): {bad_keys}. Their content was NOT delivered to the subagent. The "
        "`description` string is the subagent's ENTIRE input — there is no `prompt` or message "
        "parameter. Re-call `task` with your full request folded into `description`."
    )


def _task_arg_guard(stats: dict[str, Any], *, max_rejections: int = _MAX_TASK_ARG_REJECTIONS):
    """Lead-only middleware: reject a ``task`` call that carries unexpected kwargs (see above).

    ``stats`` is a caller-owned dict the middleware mutates so run_deepagents can report the
    outcome: ``unexpected_task_kwargs`` (the offending keys seen across the run) and
    ``task_arg_rejections`` (how many calls were bounced). It works by returning an error
    ``ToolMessage`` in place of dispatching (the same short-circuit shape the circuit breaker
    uses) — it does NOT call ``handler``, so the subagent never runs on truncated input. The
    push-back is bounded: once ``max_rejections`` is hit it RELENTS and lets the call through (the
    task tool drops the extras — the original behavior) so a stubborn model still makes progress;
    the occurrence is still recorded. Defined lazily so importing this module doesn't require
    langchain — only the deepagents driver does.
    """
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import ToolMessage

    def _reject(request):
        """Return an error ToolMessage to bounce this call, or None to dispatch normally."""
        tc = getattr(request, "tool_call", None) or {}
        if tc.get("name") != "task":
            return None
        bad = _unexpected_task_kwargs(tc.get("args"))
        if not bad:
            return None
        stats["unexpected_task_kwargs"] = sorted(
            set(stats.get("unexpected_task_kwargs") or []) | set(bad))
        if stats.get("task_arg_rejections", 0) >= max_rejections:
            print(f"[task-arg guard] unexpected task kwargs {bad} — relenting after "
                  f"{max_rejections} rejections; dispatching (the extras are dropped).")
            return None
        stats["task_arg_rejections"] = stats.get("task_arg_rejections", 0) + 1
        print(f"[task-arg guard] rejecting task(...) with unexpected kwargs {bad} "
              f"({stats['task_arg_rejections']}/{max_rejections})")
        return ToolMessage(content=_task_arg_reject_message(bad),
                           tool_call_id=tc.get("id") or "", status="error")

    class _TaskArgGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            rejection = _reject(request)
            return rejection if rejection is not None else handler(request)

        async def awrap_tool_call(self, request, handler):
            rejection = _reject(request)
            return rejection if rejection is not None else await handler(request)

    return _TaskArgGuard()


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
    bio_sessions: Any | None = None,
    bio_tools_spec: str = "",
    bio_literature: bool = False,
    curated_entities: "list[str] | None" = None,
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
    ``api_key``). Returns ``{"stop_reason": ..., "n_tool_errors": ..., "disabled_tools": ...,
    **guard_stats}`` — the stop reason, the circuit breaker's error tallies, and the
    tool-dispatch guard's counters (``tools_executed``, ``dropped_dup_tool_calls``,
    ``idle_nudges``, ``gave_up_no_tools``); the caller reads ``stop_reason`` and may ignore the
    rest. The orchestrator's messages, its ``task`` delegations, the analysts' private tool
    calls, and token usage are recorded by Phoenix tracing (set up by the runner), so this
    function just drives the loop to completion.
    """
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI

    # Adapt the MCP tools bound to this already-initialized session, then curate to the
    # union of the lead's + analysts' tools. This pool is only used to resolve each analyst's
    # tool objects; it is NOT what the lead gets (see lead_tools below).
    petab_tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
    # Blinded bio-knowledge tools from one or more SECONDARY MCP servers (biocurator joins the
    # panel on every deepagents run; endpoints hardcoded in runner.BIO_MCP_URLS = BioContextAI + OmniPath;
    # --bio-literature adds EuropePMC/bioRxiv). They are
    # RUNTIME-DISCOVERED (their names are unknown ahead of time), so they deliberately BYPASS
    # _select_tools/ALLOWED_TOOLS (which would warn+drop them as "unexpected") and are handed
    # ONLY to the biocurator subagent below — never to the lead or the other analysts. NOTE: the
    # runner's _ToolClock shadows only the PETAB session's call_tool, so bio-server latency is
    # counted as "LLM+overhead" in the metrics split rather than tool time (a known caveat now
    # that the biocurator runs by default; wrap each bio session's call_tool into the clock too
    # if that split matters).
    bio_tools: list = []
    if bio_sessions:
        petab_names = {n for t in petab_tools if (n := getattr(t, "name", None))}
        # Resolve the tool selection from the --bio-tools spec (a CURATED allowlist, not a numeric
        # cap, so we never drop the one tool the biocurator needs):
        #   ""/None  -> the curated DEFAULT_BIO_TOOLS (a lean, deliberate set spanning the default
        #               BioContextAI server + OmniPath: identity/domains/interactions/pathways/GO/
        #               ontology + OmniPath's SQL tool),
        #   "all"    -> every tool the connected server(s) expose (escape hatch),
        #   "a,b,c"  -> exactly those names.
        spec = (bio_tools_spec or "").strip()
        if spec.lower() == "all":
            allow = None
        elif spec:
            allow = {s.strip() for s in spec.split(",") if s.strip()}
        else:
            # Curated default; add the literature tools only when the operator opted in.
            allow = set(DEFAULT_BIO_TOOLS)
            if bio_literature:
                allow |= set(LITERATURE_BIO_TOOLS)
        # Discover tools across ALL bio servers, dropping any whose name COLLIDES with a petab tool
        # (would shadow it in tools_by_name for the lead/analysts) or DUPLICATES one already seen
        # from another bio server (keep the first). Bio tools keep their FULL server descriptions
        # (not _trim_descriptions, which is petab-oriented and would truncate usage/arg notes).
        discovered: list = []
        seen: set = set()
        for bs in bio_sessions:
            # Bounded + guarded: a bio server that is slow or whose (isolated) session has already
            # died must drop only its own tools, never abort the run (issue #86). A discovery
            # failure here is an ordinary Exception (the session is isolated in its own task, so it
            # never surfaces as a run-cancelling anyio group) — skip that server and continue.
            try:
                bs_tools = await asyncio.wait_for(load_mcp_tools(bs),
                                                  timeout=_BIO_DISCOVERY_TIMEOUT_S)
            except Exception as exc:
                print(f"[bio] WARNING: tool discovery on a bio server failed "
                      f"({type(exc).__name__}: {exc}); skipping that server's tools.")
                continue
            for t in bs_tools:
                nm = getattr(t, "name", None)
                if nm in petab_names:
                    print(f"[bio] WARNING: bio tool {nm!r} collides with a petab tool name; "
                          "dropping it so it cannot shadow the petab tool.")
                    continue
                if nm in seen:
                    print(f"[bio] duplicate bio tool {nm!r} from another bio server; keeping the "
                          "first.")
                    continue
                seen.add(nm)
                discovered.append((nm, t))
        names = [nm for nm, _ in discovered]
        if not discovered:
            print("[bio] the bio server(s) exposed no tools; the biocurator will run 'dumb' "
                  "(the model's own biology knowledge).")
        elif allow is None:
            bio_tools = [t for _, t in discovered]
        else:
            # No all-tools fallback: hand the biocurator ONLY the curated tools that actually
            # resolved on the connected server(s). If none match it still runs (below) — just
            # 'dumb' — and the empty biocurator_tools in result.json flags that for analysis.
            bio_tools = [t for nm, t in discovered if nm in allow]
            if not bio_tools:
                shown = f"{names[:12]}{' …' if len(names) > 12 else ''}"
                print(f"[bio] none of the curated tool names matched the connected server(s)' "
                      f"{len(names)} tools ({shown}); the biocurator will run 'dumb'.")
    all_tools = petab_tools + bio_tools
    # Wrap the tool objects with the repeated-error circuit breaker BEFORE they are handed to
    # the lead and the analyst subagents (both reference these same objects), so a tool that
    # keeps failing identically is disabled for the run instead of being re-proposed endlessly.
    breaker_state = _new_breaker_state()
    # Only the EXTERNAL bio tools (flaky hosted servers) get the immediate transport-error disable;
    # a transport blip on a core petab tool must NOT silently strip it for the rest of the run.
    _bio_names = {nm for nm in (getattr(t, "name", None) for t in bio_tools) if nm}
    _wrap_tools_with_circuit_breaker(all_tools, breaker_state, transport_disable_names=_bio_names)
    # Provenance: record the biocurator's LITERATURE accesses (query + returned DOIs/PMIDs) for
    # audit + analysis (surfaced as biocurator_literature). Only literature tools are wrapped, and
    # only when --bio-literature made them resolve into bio_tools; empty otherwise.
    literature_log: list = []
    _wrap_literature_recorder(
        [t for t in bio_tools if getattr(t, "name", None) in LITERATURE_BIO_TOOLS], literature_log
    )
    tools_by_name = {n: t for t in all_tools if (n := getattr(t, "name", None))}
    subagents = _build_subagents(tools_by_name)  # each analyst carries its own tool OBJECTS
    # The biocurator joins the panel on every deepagents run (its lack of any petab tool is the
    # structural half of the task-blinding — see _build_biocurator_subagent). It is never dropped:
    # if bio tools resolved it uses them, otherwise it runs "dumb", answering from the model's own
    # biology knowledge (see BIOCURATOR_PROMPT). The lead's roster + biocurator guidance are static
    # in scenario.DEFAULT_SYSTEM_PROMPT — no runtime prompt splice, since the biocurator is
    # unconditional. biocurator_tools records what it actually got (empty = a dumb run) for analysis.
    # Hand the biocurator the scenario's HAND-CURATED biological entity NAMES DIRECTLY (see
    # Scenario.biocurator_entities) — it holds no petab tool and cannot self-serve like the
    # analysts, so this is its reliable input channel rather than a lead `description` /
    # silently-dropped `prompt=` (issue #91). Empty for a scenario with no lookupable biology (the
    # abstract toys): the biocurator then simply gets no injected entities (nothing to look up).
    biocurator_entities = list(curated_entities or [])
    subagents.append(_build_biocurator_subagent(bio_tools, biocurator_entities))
    biocurator_tools = sorted(n for t in bio_tools if (n := getattr(t, "name", None)))
    print(f"[agent] biocurator on the panel with {len(biocurator_tools)} bio tool(s)"
          + (f": {biocurator_tools}" if biocurator_tools
             else " (none resolved — running on the model's own knowledge)"))
    print(f"[agent] biocurator handed {len(biocurator_entities)} entity name(s)"
          + (f": {biocurator_entities}" if biocurator_entities else " (none curated)"))
    # The LEAD gets ONLY the orchestrator tools (+ its decision tool) — NOT the analysts'
    # data/diagnostic tools (query_data, get_fitting_log, get_measurements, ...). Those heavy
    # reads run INSIDE analyst subagents, whose verbose results live in their own transient
    # contexts and never enter the lead's every-turn re-sent history — which is where this
    # input-dominated workload's token cost accumulates (the lead re-sends its whole history on
    # each of its 60-115 turns). The lead diagnoses by DELEGATING to the analysts, not by
    # querying data itself. deepagents' subagents hold their own resolved tool objects, so a
    # restricted top-level tool list does not starve them.
    lead_tools = [tools_by_name[n] for n in ORCHESTRATOR_TOOLS if n in tools_by_name]
    # Shared with the driver loop: the decision tool sets finish_state["confirmed"] on a
    # confirmed finish (so a legitimately-finished run is NOT bounced back), and "gave_up" after
    # too many no-fit finish attempts (so the driver ends the run promptly, not at recursion).
    finish_state = {"confirmed": False, "gave_up": False}
    lead_tools = [*lead_tools, _decision_tool(session, finish_state, probe=progress_probe)]

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_S,          # fail a hung provider call fast (was unbounded)
        max_retries=_LLM_MAX_RETRIES,
        # Ask for token usage even on streamed responses (adds stream_options.include_usage):
        # OpenAI-compatible providers omit usage from a stream unless requested, which would
        # leave _token_accountant recording 0 tokens for the whole run. Harmless when a call
        # isn't streamed.
        stream_usage=True,
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
    _TRANSIENT_ERRORS: tuple = (APIConnectionError, APITimeoutError, InternalServerError,
                                RateLimitError, JSONDecodeError)
    # Backstop: a transport error (a closed/hung MCP server) that ESCAPES the per-tool breaker —
    # e.g. raised at the `task` subagent boundary or the bio ClientSession itself — should RESUME
    # the run from the checkpoint rather than abort it (the petab server is still alive). Cover both
    # transport layers the breaker classifies in _TRANSPORT_ERROR_NAMES: the anyio stream errors and
    # httpx's TransportError base (ConnectError/ReadError/WriteError/ReadTimeout/ConnectTimeout/
    # RemoteProtocolError all derive from it).
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (ClosedResourceError, BrokenResourceError, EndOfStream)
    except Exception:  # pragma: no cover - anyio always present with mcp, but never fail on import
        pass
    try:
        import httpx
        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (httpx.TransportError,)
    except Exception:  # pragma: no cover - httpx always present with mcp, but never fail on import
        pass

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
    # filtered in this deepagents version — see _builtin_tool_filter; `task` is untouched), hide a
    # circuit-broken tool so it stops being re-proposed, and guard tool dispatch (dedupe duplicate
    # calls in a batch; push back on idle no-tool turns and label a run that never fires a tool as
    # no_tools_executed — see _tool_dispatch_guard). No history trimming: all target models have a
    # 1M-token window and the recursion limit caps a run at ~120k tokens, so the old tool-result
    # trimmer only ever fired on a rare over-long run and, with prompt caching, cost more (a clear
    # invalidates the cached prefix) than the near-free cache_read history it removed.
    guard_stats: dict[str, Any] = {"tools_executed": 0, "dropped_dup_tool_calls": 0,
                                   "idle_nudges": 0, "gave_up_no_tools": False,
                                   "unexpected_task_kwargs": [], "task_arg_rejections": 0,
                                   "subagent_names": [], "general_purpose_subagent": None}
    guard_installed = "middleware" in params
    if guard_installed:
        kwargs["middleware"] = [
            _builtin_tool_filter(), _disabled_tool_filter(breaker_state),
            # _task_arg_guard is placed OUTSIDE _tool_dispatch_guard (earlier in the list = outer
            # tool-call wrapper) so a REJECTED `task` call short-circuits before the dispatch guard
            # counts it — a bounced call never dispatched a subagent, so it must not inflate
            # tools_executed (which gates the no_tools_executed relabel / idle-nudge logic).
            _task_arg_guard(guard_stats), _tool_dispatch_guard(guard_stats),
        ]
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `middleware` parameter; "
              "the lead's built-in FS/todo tools + full history will remain, and proposed "
              "tool calls are not guarded.")
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
    # Drop the uncurated `general-purpose` subagent BEFORE the graph is built — it is added
    # during create_deep_agent, so this must precede it (issue #112). Then verify through
    # deepagents' own resolver: a run whose panel still carries it is contaminated (the lead can
    # delegate model authoring), so record the verdict in result.json rather than only logging
    # it, and warn loudly. We do NOT abort — a harness-detail drift must not cost a sweep run.
    _disable_general_purpose_subagent()
    gp_enabled = _general_purpose_enabled(llm)
    guard_stats["general_purpose_subagent"] = gp_enabled
    guard_stats["subagent_names"] = [s["name"] for s in subagents]
    if gp_enabled is not False:
        print("[agent] WARNING: deepagents' `general-purpose` subagent is "
              + ("still enabled" if gp_enabled else "of unknown status")
              + " — the lead may be able to delegate model authoring to an uncurated subagent "
                "holding the full tool pool (issue #112). Treat this run's containment as "
                "unverified.")
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
    # Sum token usage across the lead + every subagent (config callbacks propagate into the
    # subgraphs), for result.json. One instance across all resume iterations = whole-run totals.
    token_accountant = _token_accountant()
    config["callbacks"] = [token_accountant]

    async def _early_stop_reason() -> str | None:
        """The server's early-stop reason if the run should stop now, else ``None`` (fail-open:
        any probe error just means "keep going")."""
        if progress_probe is None:
            return None
        try:
            data = _structured(await progress_probe("get_progress", {}))
            if isinstance(data, dict) and data.get("should_stop"):
                return data.get("stop_reason") or "early-stop condition met"
        except Exception:  # pragma: no cover - never let the probe break the run
            pass
        return None

    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    # A silent stop (graph ends on a no-tool turn) without a confirmed finish is bounced back
    # into the finish gate this many times; separate from resume_left, which is for transients.
    silent_bounces = 0
    # Last clean-completion probe of whether a model was authored (see the silent-stop bounce);
    # captured so the post-loop relabel reuses it instead of re-probing. Only read when
    # stop_reason == "completed", which guarantees the intercept refreshed it that iteration.
    model_committed = True
    if progress_probe is None:
        # The no-model-built guard (and the deterministic early-stop) both need the probe; without
        # it a silent no-model stop scores as an empty "completed" run instead of resuming to
        # build one — so a caller that omits progress_probe reintroduces that gap silently.
        print("[agent] no progress_probe wired — the no-model-built guard and deterministic "
              "early-stop are disabled for this run")
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
                # The gate gave up (too many no-fit finish attempts) — end promptly at this
                # boundary rather than letting the model spam finish to the recursion limit.
                if finish_state["gave_up"]:
                    break
                # Deterministic early-stop: poll the server between super-steps and stop at this
                # clean boundary once the run has stopped improving (streak cap / plateau).
                early_stopped = await _early_stop_reason()
                if early_stopped is not None:
                    break
            stop_reason = (
                "wall_time_budget_exceeded" if wall_budget_exceeded
                else "no_model_built" if finish_state["gave_up"]
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
        if finish_state["gave_up"]:
            print("[agent] finish gate gave up (no fittable model after repeated attempts) — "
                  "ending the run as no-model-built.")
            break
        if retriable is None:
            # Clean graph termination. A "completed" that did NOT pass the finish gate is a SILENT
            # STOP — the lead ended on a no-tool turn, sidestepping make_decision. A silent stop is
            # not an accepted exit: bounce it back into the gate so the run only ends by finishing
            # properly (a fitted model, diagnosis shown, confirmed) — with no coaching on WHAT to
            # fix, just enforcement of the protocol. Bounded (_MAX_SILENT_BOUNCES) so a genuinely
            # stuck run still terminates at the step/wall limit rather than looping. A CONFIRMED
            # finish is respected (not bounced) and needs neither the probe nor the relabel, so it
            # short-circuits first — saving a describe_problem call on the good path. The no-model
            # case gets the build nudge; a built-but-unfinished model gets the finish nudge. No
            # backoff: a silent stop is not a rate-limited transient, so there is nothing to wait
            # out. model_committed is captured for the post-loop relabel (no second probe).
            if stop_reason == "completed" and not finish_state["confirmed"]:
                model_committed = await _model_committed(progress_probe)
                past_deadline = deadline is not None and time.monotonic() >= deadline
                if (silent_bounces < _MAX_SILENT_BOUNCES and checkpointer is not None
                        and not past_deadline):
                    silent_bounces += 1
                    nudge = _NO_MODEL_NUDGE if not model_committed else _SILENT_FINISH_NUDGE
                    what = "NO model authored" if not model_committed else "no finish through the gate"
                    print(f"[agent] silent stop ({what}) — bouncing back to finish properly "
                          f"({silent_bounces}/{_MAX_SILENT_BOUNCES})")
                    stream_input = {"messages": [{"role": "user", "content": nudge}]}
                    continue
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

    # A "completed" stop_reason that did NOT pass the finish gate (bounces exhausted, or no
    # checkpointer to resume from) is relabelled honestly rather than read as a proper finish, so
    # evaluation can see a bypassed gate. Three cases, most specific first:
    #   * no_tools_executed — the lead only ever emitted no-tool turns, so the dispatch guard
    #     pushed back _MAX_IDLE_MODEL_STEPS times and GAVE UP (guard_stats["gave_up_no_tools"]):
    #     the degenerate "proposes tool calls but none fire" run. Distinct from a run that DID
    #     dispatch tools but built no model (e.g. delegation-only), which stays no_model_built.
    #     Gated on the guard actually being installed (without the `middleware` hook the counters
    #     are not trustworthy). This must come first: gave_up_no_tools implies no model, so it
    #     would otherwise be swallowed by the no_model_built branch below.
    #   * no_model_built — nothing was authored;
    #   * silent_stop — a model exists but the run never finished through make_decision.
    # A confirmed finish keeps "completed". Reuses the intercept's probe (model_committed): a
    # "completed" stop_reason here means the final iteration ran the intercept, which refreshed it
    # — no second describe_problem call.
    if stop_reason == "completed" and not finish_state["confirmed"]:
        if guard_installed and guard_stats.get("gave_up_no_tools"):
            stop_reason = "no_tools_executed"
        else:
            stop_reason = "no_model_built" if not model_committed else "silent_stop"
    if guard_stats["dropped_dup_tool_calls"]:
        print(f"[agent] de-duplicated {guard_stats['dropped_dup_tool_calls']} identical tool "
              "call(s) within a batch over the run")
    if breaker_state["n_tool_errors"]:
        disabled = sorted(breaker_state["disabled"])
        print(f"[agent] tool errors this run: {breaker_state['n_tool_errors']}"
              + (f"; disabled: {disabled}" if disabled else ""))
    token_usage = token_accountant.totals()
    # A run that dispatched tools necessarily made LLM calls; capturing zero of them means the
    # usage was not reported (a streamed response without usage, or a callback-propagation gap),
    # so the recorded token_usage understates the run. Make that loud rather than silently
    # writing zeros that read as a genuinely tiny run in the cost analysis.
    if token_usage["n_llm_calls"] == 0 and guard_stats.get("tools_executed", 0) > 0:
        print("[agent] WARNING: token accounting captured 0 LLM calls despite tool activity — "
              "token_usage in result.json is understated (usage not reported for this run).")

    return {
        "stop_reason": stop_reason,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
        "biocurator_tools": biocurator_tools,
        "biocurator_entities": biocurator_entities,
        "biocurator_literature": literature_log,
        "token_usage": token_usage,
        **guard_stats,
    }
