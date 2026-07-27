"""Tests for the biocurator's EuropePMC payload trim (``_trim_europepmc_search`` in
``petab_harness.agent.literature``).

``get_europepmc_articles`` returns ``resultType=core`` records — for each hit the FULL
bibliographic record (author/affiliation/MeSH/grant/chemical lists, cross-references, full-text
URLs), ~50k+ tokens per search of metadata a modeller never uses. The trim keeps only the
modelling-relevant fields and drops the rest. langchain_core is a harness dep, so skip where absent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

from petab_harness.agent import literature as _literature  # noqa: E402


def _load_agent_module():
    return _literature


def _core_payload():
    """A minimal EuropePMC resultType=core response with one heavily-decorated record."""
    return json.dumps({
        "version": "6.9",
        "hitCount": 4257,
        "nextCursorMark": "AoJUUygyMzg0NDU5NQ==",
        "nextPageUrl": "https://www.ebi.ac.uk/europepmc/...",
        "request": {"queryString": "CD8 T cell kinetics", "pageSize": 10},
        "resultList": {"result": [{
            "id": "32433532", "source": "MED", "pmid": "32433532", "pmcid": "PMC7238960",
            "doi": "10.1038/s41577-020-0306-5",
            "title": "Inflammation directs memory precursor and short-lived effector CD8 T cell fates",
            "authorString": ", ".join(f"Author{i} A" for i in range(60)),  # long -> capped
            "pubYear": "2007",
            "abstractText": "Acute infection drives naive CD8 T cells to expand and differentiate.",
            "journalInfo": {"journal": {"title": "Immunity"}},
            "pubTypeList": {"pubType": ["Journal Article", "Review"]},
            "meshHeadingList": {"meshHeading": [{"descriptorName": f"M{i}"} for i in range(40)]},
            "grantsList": {"grant": [{"grantId": f"G{i}"} for i in range(25)]},
            "fullTextUrlList": {"fullTextUrl": [{"url": f"http://x/{i}"} for i in range(8)]},
            "chemicalList": {"chemical": [{"name": f"c{i}"} for i in range(15)]},
            "citedByCount": 1234,
        }]},
    })


def test_trim_drops_core_bloat_keeps_signal():
    mod = _load_agent_module()
    payload = _core_payload()
    slim = mod._trim_europepmc_search(payload)
    assert slim is not None and len(slim) < len(payload)

    d = json.loads(slim)
    r = d["resultList"]["result"][0]
    # modelling signal preserved
    assert r["doi"] == "10.1038/s41577-020-0306-5"
    assert r["pmid"] == "32433532"
    assert r["title"].startswith("Inflammation directs")
    assert r["abstractText"].startswith("Acute infection")
    assert r["journalTitle"] == "Immunity"
    assert r["pubYear"] == "2007"
    assert r["pubType"] == ["Journal Article", "Review"]  # from pubTypeList.pubType, not top level
    # bloat dropped
    for junk in ("meshHeadingList", "grantsList", "fullTextUrlList", "chemicalList",
                 "citedByCount", "journalInfo", "pubTypeList"):
        assert junk not in r
    # long author list is capped
    assert r["authorString"].endswith("…et al.")
    # top-level: hitCount + paging cursors kept (so a follow-up can page), request echo dropped
    assert d["hitCount"] == 4257
    assert d["nextCursorMark"] == "AoJUUygyMzg0NDU5NQ=="
    assert "nextPageUrl" in d
    assert "request" not in d


def test_trim_returns_none_on_unexpected_shape():
    """A non-core / unparseable payload is left untouched (the wrapper then returns it as-is)."""
    mod = _load_agent_module()
    assert mod._trim_europepmc_search("not json at all") is None
    assert mod._trim_europepmc_search(json.dumps({"foo": "bar"})) is None       # no resultList
    assert mod._trim_europepmc_search(json.dumps({"resultList": {"result": 3}})) is None  # not a list


def test_wrapper_trims_and_preserves_result_shape():
    """Drive _wrap_literature_recorder: it must record the FULL response's DOIs, hand back a
    trimmed payload, and keep the result's shape — a bare string stays a string, a
    (content, artifact) tuple keeps its artifact tail."""
    import asyncio
    mod = _load_agent_module()
    payload = _core_payload()

    class _FakeTool:
        def __init__(self, name, coro):
            self.name, self.coroutine = name, coro

    async def _string_coro(runtime=None, **kw):
        return payload                       # bare-string content

    async def _tuple_coro(runtime=None, **kw):
        return (payload, {"artifact": 1})    # (content, artifact) content

    log: list = []
    tools = [_FakeTool("get_europepmc_articles", _string_coro),
             _FakeTool("get_europepmc_articles", _tuple_coro)]
    mod._wrap_literature_recorder(tools, log)

    out_str = asyncio.run(tools[0].coroutine(runtime=None, query="q"))
    assert isinstance(out_str, str) and len(out_str) < len(payload)
    assert "meshHeadingList" not in out_str and "10.1038/s41577-020-0306-5" in out_str

    out_tup = asyncio.run(tools[1].coroutine(runtime=None, query="q"))
    assert isinstance(out_tup, tuple) and out_tup[1] == {"artifact": 1}  # artifact preserved
    assert isinstance(out_tup[0], str) and len(out_tup[0]) < len(payload)

    # both calls recorded the DOI from the FULL (untrimmed) response
    assert len(log) == 2
    assert "10.1038/s41577-020-0306-5" in log[0]["dois"]
