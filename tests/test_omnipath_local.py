"""The local OmniPath snapshot runs the SQL the biocurator writes (``gemot.omnipath_local``).

``references`` is a column of the snapshot's tables and a reserved word in DuckDB, so the module
quotes it where the model left it bare, and must leave every quoted form alone.
"""

from __future__ import annotations

import gzip

import pytest

from gemot import omnipath_local as O

_TABLES = {
    "interactions.tsv.gz": "source_genesymbol\ttarget_genesymbol\treferences\nEGFR\tGRB2\tSIGNOR:1\n",
    "enzsub.tsv.gz": "enzyme_genesymbol\tsubstrate_genesymbol\treferences\nEGFR\tSHC1\tPMID:2\n",
    "annotations.tsv.gz": "genesymbol\tsource\tlabel\tvalue\nEGFR\tSIGNOR\tpathway\tEGFR\n",
}


@pytest.fixture
def snapshot(tmp_path):
    for name, text in _TABLES.items():
        (tmp_path / name).write_bytes(gzip.compress(text.encode()))
    return tmp_path


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT references FROM interactions",
        # Run 63612603_82: rewritten to `"references"`, which DuckDB read as a column named with
        # the double quotes in it, and the tool was disabled on the errors.
        "SELECT `references` FROM interactions",
        'SELECT "references" FROM interactions',
        "SELECT i.references FROM interactions i",
    ],
)
def test_the_references_column_is_read_however_it_is_quoted(snapshot, sql):
    out = O.run_sql(sql, directory=snapshot)

    assert "error" not in out
    assert out["rows"] == [{"references": "SIGNOR:1"}]


def test_a_string_literal_is_left_alone(snapshot):
    out = O.run_sql(
        "SELECT 'references' AS label FROM interactions WHERE references = 'SIGNOR:1'",
        directory=snapshot,
    )

    assert out["rows"] == [{"label": "references"}]
