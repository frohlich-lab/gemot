"""The biocurator's UniProt entries, trimmed to what a curator reads.

BioContextAI's ``get_uniprot_protein_info`` returns the first UniProtKB search hit whole: every
sequence feature, every database cross-reference, the sequence, and with ``include_references``
every citation. TP53 (P04637) is 849 KB of JSON that way and EGFR (P00533) 385 KB, measured from
rest.uniprot.org in November 2026. The server is fastmcp 2.13, which sends a dict reply twice, as
text and as structured content, so a well-studied protein's reply passed mcp's 1 MiB limit on one
server-sent event. The client dropped it with ``SSE stream failed``, and the breaker counted
the repeats toward disabling the tool (frohlich-lab/PEtab-MCP#337, finding C25; the mechanism is
read off the code, not the run's transcript). The bio sessions now read larger events
(``bio_sessions._BIO_MAX_SSE_EVENT_BYTES``), and this module cuts the entry before the model sees
it: TP53 to 91 KB with its references (34 KB without), EGFR to 50 KB.

What stays is what a modeller of the protein's dynamics uses: its names, genes and organism, the
annotation comments (function, activity regulation, subunit, PTM, location, ...), the keywords,
and each reference's PubMed id, DOI, title, year and what UniProt cites it for. An interaction
comment keeps only the partners' gene names. What goes is the feature table (523 KB of TP53's
entry: variants, sites, secondary structure), the cross-references, the sequence (its length
stays), the audit fields and every ``evidences`` list. The comment texts cite their PubMed ids
inline, so dropping the lists loses nothing.

A separate module from :mod:`.literature`, whose trim belongs to the literature tools'
provenance recorder; this one is for a tool every biocurator holds.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

#: The tool whose replies are trimmed, BioContextAI's name for it.
UNIPROT_TOOL = "get_uniprot_protein_info"

_ORGANISM_FIELDS = ("scientificName", "commonName", "taxonId")


def _without_evidences(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_evidences(v) for k, v in value.items() if k != "evidences"}
    if isinstance(value, list):
        return [_without_evidences(v) for v in value]
    return value


def _comment(comment: dict) -> dict:
    if comment.get("commentType") != "INTERACTION":
        return _without_evidences(comment)
    partners = {
        two.get("geneName") or two.get("uniProtKBAccession")
        for i in comment.get("interactions") or []
        if isinstance(i, dict) and isinstance(two := i.get("interactantTwo"), dict)
    }
    return {"commentType": "INTERACTION", "partners": sorted(partners - {None})}


def _reference(reference: dict) -> dict:
    citation = reference.get("citation") or {}
    ids = {x.get("database"): x.get("id") for x in citation.get("citationCrossReferences") or []}
    row = {
        "pmid": ids.get("PubMed"),
        "doi": ids.get("DOI"),
        "title": citation.get("title"),
        "year": str(citation.get("publicationDate") or "")[:4],
        "cited_for": reference.get("referencePositions"),
    }
    return {k: v for k, v in row.items() if v}


def trim_uniprot_entry(text: str) -> str | None:
    """The trimmed entry as JSON, or ``None`` when ``text`` is not one UniProtKB entry (an error
    payload, say), which is then left as it is."""
    try:
        entry = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(entry, dict) or "primaryAccession" not in entry:
        return None
    organism = entry.get("organism") or {}
    slim = {
        "primaryAccession": entry["primaryAccession"],
        "uniProtkbId": entry.get("uniProtkbId"),
        "organism": {k: organism[k] for k in _ORGANISM_FIELDS if k in organism},
        "proteinDescription": _without_evidences(entry.get("proteinDescription")),
        "genes": _without_evidences(entry.get("genes")),
        "comments": [_comment(c) for c in entry.get("comments") or [] if isinstance(c, dict)],
        "keywords": [k.get("name") for k in entry.get("keywords") or [] if isinstance(k, dict)],
        "sequenceLength": (entry.get("sequence") or {}).get("length"),
    }
    if "references" in entry:
        slim["references"] = [
            _reference(r) for r in entry.get("references") or [] if isinstance(r, dict)
        ]
    return json.dumps(slim, ensure_ascii=False)


def _wrap_uniprot_trim(tools: list) -> list:
    """Trim the UniProt tool's replies in place, on the tool objects the biocurator holds.

    Layered outside the circuit breaker, which reads the reply first. The structured content in
    the artifact is replaced too: the model never reads it, but the run's state and trace keep it.
    """
    from functools import wraps

    def _make(orig):
        @wraps(orig)
        async def _trimmed(**arguments):
            result = await orig(**arguments)
            content, artifact = result if isinstance(result, tuple) else (result, None)
            text = (
                content
                if isinstance(content, str)
                else "".join(b.get("text") or "" for b in content or [] if isinstance(b, dict))
            )
            slim = trim_uniprot_entry(text)
            if slim is None:
                return result
            content = slim if isinstance(content, str) else [{"type": "text", "text": slim}]
            if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
                artifact = {**artifact, "structured_content": json.loads(slim)}
            return (content, artifact) if isinstance(result, tuple) else content

        try:  # keep the original signature: StructuredTool reads it for `callbacks` and `config`
            _trimmed.__signature__ = inspect.signature(orig)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            pass
        return _trimmed

    for tool in tools:
        orig = getattr(tool, "coroutine", None)
        if getattr(tool, "name", None) == UNIPROT_TOOL and orig is not None:
            tool.coroutine = _make(orig)
    return tools
