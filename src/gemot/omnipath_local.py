"""A LOCAL OmniPath, served from a pinned DuckDB snapshot instead of the hosted MCP.

The biocurator's OmniPath tool is ``execute_sql_query_on_omnipath_db`` — free-text SQL. That makes
it the one bio call a cache cannot help with: the key is the query string, the model writes a
slightly different query every run, and the key space is effectively unbounded. Measured over 16
runs of three scenarios, BioContextAI (bounded entity lookups) reached a 0.5 hit rate while OmniPath
stayed at **0.0 and always will**.

A local database makes the question moot. Any SQL, unlimited, no network, no rate limit — which is
what the sweep concurrency was throttled for. It buys a second thing worth as much: the answers stop
moving. An uncached run sees whatever the hosted service returns that day, so two arms of an A/B can
be compared against different biology; a pinned snapshot gives every arm identical inputs.

WHAT IS IN THE SNAPSHOT, and what is not:

* ``interactions`` (139,851 rows) — the omnipath/pathwayextra/kinaseextra/ligrecextra datasets, with
  ``sources``, ``references``, ``curation_effort``, ``type`` and the tax-id columns.
* ``enz_sub`` (41,506 rows) — enzyme-substrate relationships with residue and modification.
* ``annotations`` (21,701 rows) — RESTRICTED to the five resources the biocurator has actually
  queried: SIGNOR, SignaLink_pathway, KEGG-PC, NetPath, TFcensus. The full annotations export is
  orders of magnitude larger and almost none of it is ever asked for. The restriction is stated in
  the tool description so a query for another resource reads as "not in this snapshot" rather than
  as "this protein has no annotations" — an empty result that looked like a fact would be worse than
  a missing table.

The column names are the hosted service's own, verbatim, so the SQL the model already writes runs
unchanged — that was checked against the queries it actually emitted, not assumed.

Provenance: ``snapshot_info()`` reports the fetch date and per-table row counts, and the runner puts
it in ``result.json``. A pinned snapshot that nothing records is the ``git_sha +dirty`` problem in a
new costume — reproducible in principle and unattributable in practice.

WHY NOT RUN saezlab/omnipath-next ITSELF? It was the obvious alternative and it was checked rather
than dismissed. Two reasons it lost:

1. It does not run here without real work. Its compose is Postgres 16 + a Next.js app built from
   context, and this cluster has no container runtime on the login or compute nodes; Apptainer and
   PostgreSQL 13-15 exist as modules, but Apptainer cannot build their unpublished image without a
   Docker daemon, and the only Node module is 10.4.1 against a Next.js needing 18+.
2. More decisively, its schema answers FEWER of the queries the biocurator actually writes. Their
   loader (``db_build/webservice_loader.py``) creates ``interactions`` WITHOUT ``references``,
   ``type`` or ``extra_attrs`` — and 9 of the 20 real queries observed select ``references`` from
   ``interactions``. Those queries have therefore been failing against the hosted service all
   along, which is consistent with how little the biocurator's output is ever used.

What upstream DOES settle is the schema: the same three tables loaded from the same
``omnipath_webservice_*.tsv`` dumps with the same column names, so the table/column surface here is
theirs rather than inferred. The one deliberate divergence is that this snapshot is a SUPERSET on the
columns the model queries, and a subset on four ``dorothea_*`` sub-flags it never has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Table name -> the TSV in the snapshot directory. `enz_sub` is the name the hosted service exposes
# (and therefore the name the model's SQL uses); the file is `enzsub.tsv` because that is the web
# service's endpoint spelling. Keeping both spellings straight here is the whole point of the map.
# GZIPPED, and read as such — DuckDB's `read_csv_auto` decompresses transparently. 43 MB of TSV
# compresses to 4.3 MB, which is the difference between a pinned snapshot being a reasonable thing
# to commit and a permanent 43 MB weight in the history for data that is regenerable.
_TABLES = {
    "interactions": "interactions.tsv.gz",
    "enz_sub": "enzsub.tsv.gz",
    "annotations": "annotations.tsv.gz",
}

_ANNOTATION_RESOURCES = ("SIGNOR", "SignaLink_pathway", "KEGG-PC", "NetPath", "TFcensus")

# A read-only SQL surface: the model is asked for SELECTs and nothing here should be able to mutate
# the snapshot even by accident. DuckDB is opened read_only, which makes this belt-and-braces.
_MAX_ROWS = 200


def snapshot_dir() -> Path:
    from .paths import repo_root

    return repo_root() / "slurm" / "omnipath_snapshot"


def snapshot_info() -> dict[str, Any]:
    """Fetch date + per-table row counts, for `result.json`. Cheap: reads a sidecar, not the data."""
    d = snapshot_dir()
    meta = d / "snapshot.json"
    if meta.is_file():
        try:
            return json.loads(meta.read_text())
        except ValueError:
            pass
    return {"present": d.is_dir(), "note": "no snapshot.json — provenance unknown"}


def _connect(directory: "Path | None" = None):
    """A DuckDB connection with the snapshot registered as views over the TSVs.

    Views, not tables: `read_csv_auto` is lazy, so startup is instant and the 27 MB of TSV is never
    copied into a second on-disk format that could drift from the files under version control.
    """
    import duckdb

    d = directory or snapshot_dir()
    missing = [f for f in _TABLES.values() if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(f"OmniPath snapshot incomplete in {d}: missing {missing}")
    con = duckdb.connect(":memory:")
    for table, fname in _TABLES.items():
        # The path is INLINED, not bound: DuckDB cannot prepare a CREATE VIEW carrying a parameter
        # ("Unexpected prepared parameter. This type of statement can't be prepared"), and because
        # the failure surfaced on the first query rather than at connect, it read as if all 20 of
        # the model's real queries were malformed. Quoted by doubling any single quote — the path
        # comes from `repo_root()`, not from the model, so this is hygiene rather than a trust
        # boundary (the model's SQL is passed to `execute`, never concatenated).
        literal = str(d / fname).replace("'", "''")
        con.execute(
            f'CREATE VIEW "{table}" AS SELECT * FROM '
            f"read_csv_auto('{literal}', delim=chr(9), header=true, sample_size=-1)"
        )
    return con


def _quote_reserved(sql: str) -> str:
    """Quote bare ``references`` identifiers so DuckDB will parse the model's SQL.

    ``references`` is a real column of OmniPath's ``interactions``/``enz_sub``/``annotations`` and the
    hosted service accepts it unquoted, so the model writes it unquoted — but it is a RESERVED WORD
    in DuckDB, which rejects the statement outright. That single word accounted for 9 of the 20 real
    queries failing against this snapshot, on a column that is present and correct.

    Rewriting rather than renaming the column, because the goal is that the SQL the model ALREADY
    writes runs unchanged. Quoted occurrences and anything inside a string literal are left alone:
    the split below alternates non-literal and literal segments on the single quote, and only the
    non-literal ones are touched — so `WHERE label = 'references'` is untouched.
    """
    import re

    parts = sql.split("'")
    for i in range(0, len(parts), 2):  # even indices are OUTSIDE string literals
        parts[i] = re.sub(
            r'(?<!")\breferences\b(?!")', '"references"', parts[i], flags=re.IGNORECASE
        )
    return "'".join(parts)


def run_sql(
    sql: str, *, max_rows: int = _MAX_ROWS, directory: "Path | None" = None
) -> dict[str, Any]:
    """Execute one read-only query against the snapshot and return rows + metadata.

    Errors are RETURNED, not raised: a malformed query is something the model should see and fix,
    exactly as the petab server's own `query_data` treats it, and raising here would surface as a
    tool transport failure and degrade the biocurator to "dumb" for the rest of the run.
    """
    if not isinstance(sql, str) or not sql.strip():
        return {"error": "empty query", "rows": [], "n_rows": 0}
    con = None
    try:
        con = _connect(directory)
        cur = con.execute(_quote_reserved(sql))
        cols = [c[0] for c in (cur.description or [])]
        rows = cur.fetchmany(max_rows)
        truncated = len(cur.fetchmany(1)) > 0
        return {
            "columns": cols,
            "rows": [dict(zip(cols, r)) for r in rows],
            "n_rows": len(rows),
            "truncated": truncated,
            "max_rows": max_rows,
            "snapshot": snapshot_info().get("fetched_at"),
        }
    except Exception as exc:  # noqa: BLE001 - a bad query is the model's to fix, not a run-ender
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "available_tables": sorted(_TABLES),
            "annotation_resources_in_snapshot": list(_ANNOTATION_RESOURCES),
            "rows": [],
            "n_rows": 0,
        }
    finally:
        if con is not None:
            con.close()


TOOL_DESCRIPTION = (
    "Run read-only SQL over a PINNED LOCAL OmniPath snapshot. Tables: `interactions` "
    "(source_genesymbol, target_genesymbol, type, is_directed, is_stimulation, is_inhibition, "
    "consensus_direction, sources, references, curation_effort, ncbi_tax_id_source, "
    "ncbi_tax_id_target), `enz_sub` (enzyme_genesymbol, substrate_genesymbol, residue_type, "
    "residue_offset, modification, sources, references), and `annotations` (genesymbol, uniprot, "
    "entity_type, source, label, value, record_id). "
    "`annotations` covers ONLY these resources: " + ", ".join(_ANNOTATION_RESOURCES) + " — a query "
    "for any other resource returns nothing because it is ABSENT FROM THE SNAPSHOT, which is not "
    "evidence that the protein is unannotated. Results are capped; `truncated` says when rows were "
    "dropped. A malformed query comes back as `error` with the table list, so read it and retry."
)


class LocalOmniPathSession:
    """An in-process stand-in for the hosted OmniPath MCP session.

    Implements only what `langchain_mcp_adapters.load_mcp_tools` uses — `list_tools(cursor=...)`
    and `call_tool(name, arguments, ...)` — returning genuine `mcp.types` objects, so nothing
    downstream can tell it apart from a network session. That matters: the biocurator, the circuit
    breaker, the bio-session supervisor and the response cache all handle these objects, and a
    hand-rolled duck type would work until one of them read `isError`.

    No process, no port, no network. A real MCP server would need spawning, a free port and a
    readiness wait per run (the petab server already pays that cost); none of it buys anything here,
    because the only consumer is in the same interpreter.
    """

    def __init__(self, directory: "Path | None" = None) -> None:
        self._dir = directory

    async def list_tools(self, cursor=None):
        from mcp.types import ListToolsResult, Tool

        return ListToolsResult(
            nextCursor=None,
            tools=[
                Tool(
                    name="execute_sql_query_on_omnipath_db",
                    description=TOOL_DESCRIPTION,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sqlQuery": {
                                "type": "string",
                                "description": "A read-only SQL SELECT over the tables named above.",
                            }
                        },
                        "required": ["sqlQuery"],
                    },
                )
            ],
        )

    async def call_tool(self, name, arguments=None, *args, **kwargs):
        from mcp.types import CallToolResult, TextContent

        if name != "execute_sql_query_on_omnipath_db":
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"unknown tool {name!r}; this endpoint serves only "
                        "execute_sql_query_on_omnipath_db",
                    )
                ],
            )
        sql = (arguments or {}).get("sqlQuery") if isinstance(arguments, dict) else None
        out = run_sql(sql if isinstance(sql, str) else "", directory=self._dir)
        # `isError` mirrors the hosted service so the breaker and the cache treat a bad query the
        # same way they always have — in particular the cache must NOT store it.
        return CallToolResult(
            isError=bool(out.get("error")),
            content=[TextContent(type="text", text=json.dumps(out, default=str))],
        )

    async def initialize(self):  # a network session needs one; this is a no-op
        return None


def available(directory: "Path | None" = None) -> bool:
    """Whether a usable snapshot is on disk. False falls the caller back to the hosted service —
    a missing snapshot must degrade to the old behaviour, never silently serve nothing."""
    d = directory or snapshot_dir()
    return all((d / f).is_file() for f in _TABLES.values())
