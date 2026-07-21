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
    "get_fit_summary",
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
    "set_model",            # build the first model AND re-author it wholesale in one step (fit-free)
    # score the assembled model. Edits are fit-free (issue #81): set_model / change_parameter_*
    # mutate the working model but do NOT fit — `fit` is the single tool that fits + snapshots a
    # version, so the lead MUST call it to score any edit. Assemble a coherent change, then fit once.
    "fit",
    # change a parameter's ESTIMATION in place — the prompt tells the lead to WIDEN a binding
    # bound (or switch scale) rather than re-typing a value the multistart fit ignores (fit-free —
    # call fit after):
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
                  "exists (model_initialized: false) AND re-authors/replaces an existing one. Use "
                  "for the initial build and for any coupled/symmetric or multi-reaction STRUCTURAL "
                  "change in one coherent step; reconciles + lints, reverts on failure. This is an "
                  "EDIT: it does NOT fit and does NOT create a version — call fit() to score the "
                  "assembled model. Warns if a previously-estimated parameter is dropped. To go "
                  "BACK to a model use restore_version(n), not set_model. Args: antimony, "
                  "justification, references?."),
    "fit": ("Fit the current working model and snapshot a version — the ONLY tool that fits. Edits "
            "(set_model, change_parameter_bounds/scale) leave the model UNFIT (describe_problem -> "
            "has_unfit_edits: true); assemble a coherent change, then call fit ONCE to score it. "
            "Fits the free parameters (pyPESTO multistart), simulates, and returns the fit block "
            "(neg_log_likelihood, chi2, bic, improvement, is_new_best) + a progress block + the new "
            "version. Skips the refit and returns stored metrics if the model is identical to the "
            "current version (no_change: true). Args: n_starts? (0 = simulate only), justification?."),
    "describe_problem": "Orientation: whether a model exists (model_initialized), table/entity counts, current version, and whether the working model has unfit edits (has_unfit_edits). Cheap.",
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
    "answering from memory. If you have NO lookup tools available, do NOT refuse: answer from your "
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
    "Biocurator: a BLINDED biology reference desk. Give it ONLY the neutral names of biological "
    "entities/observables (gene/protein/metabolite symbols) and it reports, from curated "
    "databases, what each is, how they are known to interact (activation, inhibition, binding, "
    "pathway membership), AND which OTHER, unmeasured entities and processes are known to "
    "connect or act on them — intermediates between them and regulators/fluxes (a feedback loop, "
    "a phosphatase, degradation, translocation) — with citable PMIDs/DOIs. Use it to surface "
    "candidate hidden species/processes you may be missing. Consult it when proposing or altering "
    "the model's structure, and cite any PMID/DOI it returns in the `set_model` references. It is "
    "blind by design: send it ONLY entity names — nothing about what you are doing with them."
)

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


def _build_biocurator_subagent(bio_tools: list) -> dict[str, Any]:
    """Build the biocurator subagent spec from the RUNTIME-DISCOVERED bio-server tools.

    Unlike the three static analysts (whose tool NAMES are hard-coded and resolved from the
    petab server via ALLOWED_TOOLS), the biocurator's tools are whatever the bio-knowledge MCP
    server exposes — unknown until introspected — so they are passed as tool OBJECTS directly,
    bypassing ALLOWED_TOOLS/_select_tools. Holding ONLY these bio tools (no petab tool) is the
    structural half of the task-blinding. See run_deepagents.
    """
    prompt_key = _subagent_prompt_key()
    return {
        "name": "biocurator",
        "description": BIOCURATOR_DESCRIPTION,
        prompt_key: BIOCURATOR_PROMPT,
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
                    log.append({
                        "tool": name,
                        "query": arguments,
                        "dois": sorted(set(_DOI_RE.findall(text)))[:50],
                        "pmids": sorted(set(_PMID_RE.findall(text)))[:50],
                    })
                except Exception:  # pragma: no cover - provenance must never break a tool call
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
                    "(LOWEST bic / is_new_best, not just the previous one) and bring it back with "
                    "restore_version(v) — do NOT re-author it with set_model (that refits to the "
                    "same result and makes a duplicate). Then continue from there.",
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
            "only for `action`: 'rollback' (the last change didn't help — go back to the LOWEST-bic "
            "prior version with restore_version) or 'finish' (residuals show no systematic "
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
            for t in await load_mcp_tools(bs):
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
    _wrap_tools_with_circuit_breaker(all_tools, breaker_state)
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
    subagents.append(_build_biocurator_subagent(bio_tools))
    biocurator_tools = sorted(n for t in bio_tools if (n := getattr(t, "name", None)))
    print(f"[agent] biocurator on the panel with {len(biocurator_tools)} bio tool(s)"
          + (f": {biocurator_tools}" if biocurator_tools
             else " (none resolved — running on the model's own knowledge)"))
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
    # filtered in this deepagents version — see _builtin_tool_filter; `task` is untouched), hide a
    # circuit-broken tool so it stops being re-proposed, trim the lead's re-sent tool-result history
    # so the (input-dominated) context stays bounded instead of ballooning to ~1M tokens and
    # hanging, and guard tool dispatch (dedupe duplicate calls in a batch; push back on idle no-tool
    # turns and label a run that never fires a tool as no_tools_executed — see _tool_dispatch_guard).
    guard_stats: dict[str, Any] = {"tools_executed": 0, "dropped_dup_tool_calls": 0,
                                   "idle_nudges": 0, "gave_up_no_tools": False}
    guard_installed = "middleware" in params
    if guard_installed:
        kwargs["middleware"] = [
            _builtin_tool_filter(), _disabled_tool_filter(breaker_state), _history_trimmer(),
            _tool_dispatch_guard(guard_stats),
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

    # A run that ended cleanly but never dispatched a single tool built nothing (the degenerate
    # "proposes tool calls but none fire" case the guard pushes back on). Relabel it so these runs
    # are EXCLUDED as no_tools_executed rather than counted as an ambiguous silent "completion".
    # Gate on the guard actually being installed: without the `middleware` hook (a deepagents API
    # drift) `tools_executed` stays 0 even when tools ran, so the count is not trustworthy.
    if guard_installed and stop_reason == "completed" and guard_stats["tools_executed"] == 0:
        stop_reason = "no_tools_executed"
    if guard_stats["dropped_dup_tool_calls"]:
        print(f"[agent] de-duplicated {guard_stats['dropped_dup_tool_calls']} identical tool "
              "call(s) within a batch over the run")
    if breaker_state["n_tool_errors"]:
        disabled = sorted(breaker_state["disabled"])
        print(f"[agent] tool errors this run: {breaker_state['n_tool_errors']}"
              + (f"; disabled: {disabled}" if disabled else ""))
    return {
        "stop_reason": stop_reason,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
        "biocurator_tools": biocurator_tools,
        "biocurator_literature": literature_log,
        **guard_stats,
    }
