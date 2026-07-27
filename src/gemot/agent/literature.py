"""Provenance for what the biocurator READ, and a trim for what it gets back.

Two jobs on the (opt-in) literature tools: record every search — the query and the DOIs/PMIDs it
returned — into a run-scoped log surfaced as ``biocurator_literature`` in ``result.json``, and
strip EuropePMC's ``resultType=core`` records down to the modelling-relevant fields before they
reach the model's context.
"""

from __future__ import annotations

from typing import Any
import inspect
import re

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
