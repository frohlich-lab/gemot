"""The blinded fourth analyst: a biology reference desk over external knowledge MCP servers.

Its tools are runtime-discovered from the bio servers (their names are unknown ahead of time), so
unlike the three static analysts it is assembled here from tool OBJECTS rather than declared in
:data:`.subagents.ANALYSTS`. Its persona — the other half of the task blinding — is
:data:`petab_harness.prompts.BIOCURATOR_PROMPT`.
"""

from __future__ import annotations

from typing import Any

from ..prompts import (BIOLOGY_BLIND_ENABLED, BIOCURATOR_DESCRIPTION, BIOCURATOR_PROMPT,
                       _biocurator_entity_block)
from .guards import _reasoning_runaway_guard, _subagent_turn_limit
from .subagents import ACTIVE_SPECS, _subagent_prompt_key

#: Model calls one biocurator delegation may make before it must report.
#:
#: The same 20 as the analysts, on measurement rather than reasoning. The prior guess here was
#: that being handed the whole entity list at once would make it the LONGEST-running subagent
#: and earn it a bigger budget; across 246 delegations in the main sweep it is in fact one of
#: the SHORTEST (med 2, p95 10, max 17) -- it batches lookups instead of serialising them.
#: 20 clears its observed maximum outright, so this cap never truncates a real survey.
BIOCURATOR_MAX_TURNS = 20

# ---------------------------------------------------------------- biocurator (optional)
# A FOURTH analyst, added for the deepagents driver whenever a bio-knowledge server is
# reachable. Unlike the three above -- which reason over THIS problem's model/data through
# the petab-mcp tools -- the biocurator reaches OUTSIDE the problem, into curated biomedical
# knowledge bases served by a SEPARATE bio-knowledge MCP server (e.g. BioContextAI:
# STRING/KEGG/Reactome/OpenTargets + literature). Its tools are runtime-discovered from that
# server (their names are unknown ahead of time), so it is built dynamically in
# run_deepagents rather than listed in ANALYSTS.
#
# It is deliberately BLINDED to the modeling task: it holds the bio tools plus exactly ONE petab
# read — `list_observables`, whose `observableName` states what each assay measures in
# biology-only terms (phosphosites, proteoforms, assay type): the vocabulary a curator searches
# with, and a surface the blinding already curates. Beyond that, never a petab
# tool that would reveal the data/task (the STRUCTURAL half of the blind; the persona half is
# prompts.BIOCURATOR_PROMPT, which never mentions a model, data, fitting or an experiment).

BIOCURATOR_NAME = "biocurator"


# The COMPLETE panel: the only `subagent_type` values the lead may delegate to. deepagents
# additionally AUTO-INSERTS a `general-purpose` subagent holding the FULL tool pool unless the
# active harness profile disables it — which let the lead delegate `set_model`/`fit`/
# `change_parameters` into a subagent and bypass lead-only authoring, plus sink ~39% of that
# subagent's calls into unused filesystem builtins (issue #112). run_deepagents turns that
# auto-add off (see _disable_general_purpose_subagent), so this tuple is the whole registry.
SUBAGENT_NAMES = (*(spec["name"] for spec in ACTIVE_SPECS),
                 *((BIOCURATOR_NAME,) if not BIOLOGY_BLIND_ENABLED else ()))

# Curated default tool allowlist for the biocurator. The default bio server (BioContextAI
# Knowledgebase) exposes ~50 tools across ~20 databases — literature search, clinical trials,
# drug labels, grants, cell-type markers, structures, … — and every tool's JSON schema is re-sent
# to the model on EVERY turn (the per-turn input cost the petab analysts are curated against). A
# numeric cap would be blunt (it can drop the very tool needed), so instead we hand the biocurator
# a small, DELIBERATE set covering its remit: entity identity + protein domains + interactions +
# pathways + GO biological processes (for latent processes/regulation). Deliberately excluded:
# free-text literature (europepmc/biorxiv/scholar — also the leak vector), clinical/drug/grants,
# AlphaFold/PRIDE/PanglaoDB/AntibodyRegistry. These are the exact tool names the BioContextAI
# server registers (v0.2.x). If NONE of these match the connected server (a different bio
# server), the biocurator runs 'dumb' with a warning (see run_deepagents) rather than being
# handed every tool the server exposes.
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
    # back to this tuple if you point at a server that has valid KEGG access.)
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
# opts in with PETAB_BIO_LITERATURE=1. Kept OUT of the default because free-text literature search over
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


async def discover_bio_tools(bio_sessions, *, literature: bool,
                             exclude_names: "set[str] | frozenset[str]" = frozenset(),
                             holder: str = "the biocurator") -> list:
    """The curated bio tools that resolved on the connected server(s), as tool objects.

    ``DEFAULT_BIO_TOOLS`` plus ``LITERATURE_BIO_TOOLS`` when ``literature`` — an allowlist of
    NAMES, not a numeric cap, so it can never drop the one tool the holder needs. Per-server
    discovery is bounded and guarded (issue #86): a slow or already-dead server costs only its
    own tools. A name in ``exclude_names`` (a petab tool it would shadow in ``tools_by_name``) or
    a duplicate from a second server is dropped with a warning. Tools keep their FULL server
    descriptions — ``_trim_descriptions`` is petab-oriented and would truncate usage notes.

    Nothing resolved returns ``[]``: there is no all-tools fallback, and ``holder`` then runs
    "dumb" on the model's own biology knowledge, which the empty ``biocurator_tools`` in
    ``result.json`` flags for analysis.
    """
    import asyncio

    from langchain_mcp_adapters.tools import load_mcp_tools

    if not bio_sessions:
        return []
    allow = set(DEFAULT_BIO_TOOLS)
    if literature:
        allow |= set(LITERATURE_BIO_TOOLS)
    discovered: list = []
    seen: set = set()
    for bs in bio_sessions:
        try:
            bs_tools = await asyncio.wait_for(load_mcp_tools(bs), timeout=_BIO_DISCOVERY_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - one server's failure must not end the run
            print(f"[bio] WARNING: tool discovery on a bio server failed "
                  f"({type(exc).__name__}: {exc}); skipping that server's tools.")
            continue
        for t in bs_tools:
            nm = getattr(t, "name", None)
            if nm in exclude_names:
                print(f"[bio] WARNING: bio tool {nm!r} collides with a petab tool name; "
                      "dropping it so it cannot shadow the petab tool.")
                continue
            if nm in seen:
                print(f"[bio] duplicate bio tool {nm!r} from another bio server; keeping the first.")
                continue
            seen.add(nm)
            discovered.append((nm, t))
    names = [nm for nm, _ in discovered]
    if not discovered:
        print(f"[bio] the bio server(s) exposed no tools; {holder} will run 'dumb' "
              "(the model's own biology knowledge).")
        return []
    bio_tools = [t for nm, t in discovered if nm in allow]
    if not bio_tools:
        shown = f"{names[:12]}{' …' if len(names) > 12 else ''}"
        print(f"[bio] none of the curated tool names matched the connected server(s)' "
              f"{len(names)} tools ({shown}); {holder} will run 'dumb'.")
    return bio_tools


def _build_biocurator_subagent(bio_tools: list, entity_names: "list[str] | None" = None,
                               stats: "dict[str, Any] | None" = None,
                               model: "Any | None" = None) -> dict[str, Any]:
    """Build the biocurator subagent spec from the RUNTIME-DISCOVERED bio-server tools.

    Unlike the three static analysts (whose tool NAMES are hard-coded and resolved from the
    petab server via ALLOWED_TOOLS), the biocurator's tools are whatever the bio-knowledge MCP
    server exposes — unknown until introspected — so they are passed as tool OBJECTS directly,
    bypassing ALLOWED_TOOLS/_select_tools. Holding no petab tool beyond the read-only
    `list_observables` (appended by the caller; owner decision 2026-08-26 — the observable
    DESCRIPTIONS are curated biology, and reading them beats relaying entities by hand) is the
    structural half of the task-blinding. See run_deepagents.

    ``model`` puts this subagent on its own chat model rather than the lead's — the biocurator is
    the panel's heaviest reader (entity lookups return whole database entries) and its job is
    retrieval, not modelling, so it is the seat where a cheaper model is worth measuring.

    ``entity_names`` (the hand-curated biological entities — see Scenario.biocurator_entities) is
    injected into the persona prompt so the biocurator is handed its input DIRECTLY rather than
    relying on the lead relaying it through the ``task`` ``description`` (issue #91). ``None``/
    empty leaves the prompt unchanged (no entities to look up — e.g. an abstract problem).
    """
    prompt_key = _subagent_prompt_key()
    spec: dict[str, Any] = {
        "name": BIOCURATOR_NAME,
        "description": BIOCURATOR_DESCRIPTION,
        prompt_key: BIOCURATOR_PROMPT + _biocurator_entity_block(entity_names or []),
        "tools": list(bio_tools),
        # Bounded like the analysts, and for the same reason -- but a touch longer: the
        # biocurator makes one external lookup per entity, and it is handed the whole entity
        # list at once, so its floor scales with that list rather than being a single question.
        # Runaway-guarded like the analysts, and for the same measured reason -- 3 of the 11
        # subagent-side truncations were the biocurator's (see _build_subagents).
        "middleware": ([_subagent_turn_limit(BIOCURATOR_MAX_TURNS)]
                       + ([_reasoning_runaway_guard(stats)] if stats is not None else [])),
    }
    # deepagents reads `model` off the spec and falls back to the lead's, so an unset key is the
    # single-model panel unchanged (``spec.get("model", model)`` in its graph builder).
    if model is not None:
        spec["model"] = model
    return spec
