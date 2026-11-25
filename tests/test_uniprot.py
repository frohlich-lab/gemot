"""The biocurator's UniProt entries reach it trimmed (``gemot.agent.uniprot``).

TP53's entry, whole, is 849 KB of JSON and was sent twice in one reply, past mcp's 1 MiB limit on
one server-sent event (frohlich-lab/PEtab-MCP#337). The entry here has the real shape, cut down.
"""

from __future__ import annotations

import asyncio
import json

from gemot.agent import uniprot as U

_EVIDENCE = [{"evidenceCode": "ECO:0000269", "source": "PubMed", "id": "12810724"}]

_ENTRY = {
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "primaryAccession": "P04637",
    "secondaryAccessions": ["Q15086"],
    "uniProtkbId": "P53_HUMAN",
    "entryAudit": {"entryVersion": 300},
    "organism": {
        "scientificName": "Homo sapiens",
        "commonName": "Human",
        "taxonId": 9606,
        "lineage": ["Eukaryota", "Metazoa"],
    },
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}},
    },
    "genes": [{"geneName": {"value": "TP53"}, "synonyms": [{"value": "P53"}]}],
    "comments": [
        {
            "commentType": "FUNCTION",
            "texts": [{"evidences": _EVIDENCE, "value": "Acts as a tumor suppressor."}],
        },
        {
            "commentType": "INTERACTION",
            "interactions": [
                {
                    "interactantOne": {"uniProtKBAccession": "P04637"},
                    "interactantTwo": {"uniProtKBAccession": "Q00987", "geneName": "MDM2"},
                    "numberOfExperiments": 30,
                },
                {
                    "interactantOne": {"uniProtKBAccession": "P04637"},
                    "interactantTwo": {"uniProtKBAccession": "P00519", "geneName": "ABL1"},
                    "numberOfExperiments": 2,
                },
            ],
        },
    ],
    "features": [{"type": "Natural variant", "location": {"start": {"value": 72}}}],
    "keywords": [{"id": "KW-0043", "category": "Molecular function", "name": "Tumor suppressor"}],
    "references": [
        {
            "referenceNumber": 1,
            "citation": {
                "id": "4006916",
                "citationType": "journal article",
                "authors": ["Zakut-Houri R."],
                "citationCrossReferences": [
                    {"database": "PubMed", "id": "4006916"},
                    {"database": "DOI", "id": "10.1002/j.1460-2075.1985.tb03768.x"},
                ],
                "title": "Human p53 cellular tumor antigen.",
                "publicationDate": "1985",
                "journal": "EMBO J.",
            },
            "referencePositions": ["NUCLEOTIDE SEQUENCE [MRNA]"],
        }
    ],
    "uniProtKBCrossReferences": [{"database": "PDB", "id": "1A1U"}],
    "sequence": {"value": "MEEPQSDPSV", "length": 393},
}


def test_an_entry_keeps_what_a_curator_reads():
    slim = json.loads(U.trim_uniprot_entry(json.dumps(_ENTRY)))

    assert slim == {
        "primaryAccession": "P04637",
        "uniProtkbId": "P53_HUMAN",
        "organism": {"scientificName": "Homo sapiens", "commonName": "Human", "taxonId": 9606},
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
        },
        "genes": [{"geneName": {"value": "TP53"}, "synonyms": [{"value": "P53"}]}],
        "comments": [
            {"commentType": "FUNCTION", "texts": [{"value": "Acts as a tumor suppressor."}]},
            {"commentType": "INTERACTION", "partners": ["ABL1", "MDM2"]},
        ],
        "keywords": ["Tumor suppressor"],
        "sequenceLength": 393,
        "references": [
            {
                "pmid": "4006916",
                "doi": "10.1002/j.1460-2075.1985.tb03768.x",
                "title": "Human p53 cellular tumor antigen.",
                "year": "1985",
                "cited_for": ["NUCLEOTIDE SEQUENCE [MRNA]"],
            }
        ],
    }


def test_an_entry_asked_for_without_references_gets_none():
    """The server drops `references` unless `include_references` is set; the trim adds none."""
    entry = {k: v for k, v in _ENTRY.items() if k != "references"}

    assert "references" not in json.loads(U.trim_uniprot_entry(json.dumps(entry)))


def test_anything_but_an_entry_is_left_alone():
    for text in ('{"error": "No results found for the given query."}', "not json", "[]"):
        assert U.trim_uniprot_entry(text) is None


class _Tool:
    def __init__(self, name, reply):
        self.name = name

        async def _run(**arguments):
            return reply

        self.coroutine = _run


def test_the_wrapped_tool_returns_the_trimmed_entry_in_text_and_artifact():
    reply = ([{"type": "text", "text": json.dumps(_ENTRY)}], {"structured_content": _ENTRY})
    uniprot, other = _Tool(U.UNIPROT_TOOL, reply), _Tool("get_string_interactions", reply)
    U._wrap_uniprot_trim([uniprot, other])

    content, artifact = asyncio.run(uniprot.coroutine(gene_symbol="TP53"))

    assert json.loads(content[0]["text"]) == artifact["structured_content"]
    assert "features" not in artifact["structured_content"]
    assert asyncio.run(other.coroutine()) is reply


def test_an_error_reply_reaches_the_curator_as_it_was():
    reply = ([{"type": "text", "text": '{"error": "No results found."}'}], None)
    tool = _Tool(U.UNIPROT_TOOL, reply)
    U._wrap_uniprot_trim([tool])

    assert asyncio.run(tool.coroutine(gene_symbol="NOPE")) is reply
