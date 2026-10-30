"""What ``archive_run`` must carry out of a work dir before the next run wipes it.

The archive is the only durable copy of a finished run's on-disk half (the Phoenix trace holds
the conversation; ``work/`` is ``rmtree``d by the next run of the same example). These tests pin
WHICH files that is — in particular ``errors.jsonl``, the record of what the server REFUSED,
which every editing A/B's rejection-class analysis reads and which used to be reconstructible
only from live work dirs; and each version's statements and multistart, which a sweep task's
cleanup took with it until #62.
"""

from __future__ import annotations

import gzip
import json

from gemot.analysis import read_version_json
from gemot.explorer import bundle as B
from gemot.explorer.archive import archive_run

# What a 3-start fit leaves in optimization.json: two starts reach the best fval (within the
# server's 1e-3), one stops on the time cap.
_OPTIMIZATION = {
    "n_starts": 3,
    "starts": [
        {"start": i, "id": str(i), "fval": f, "time": 1.5, "n_fval": 9, "x": [0.1 * i, 2.0]}
        for i, f in enumerate([4.0, 4.0005, 9.0])
    ],
    "trace": [{"start": 0, "iteration": 0, "fval": 50.0, "time": 0.1}],
    "x_names": ["k1", "sigma"],
    "x_scales": ["log10", "lin"],
    "exitflag_counts": {"ExitFlag.FTOL": 2, "ExitFlag.MAXTIME": 1},
}

# A record in the shape stored before frohlich-lab/PEtab-MCP#135 gave each row's binding apart
# (`policy` on the row, no `bindings`): kept so, since the archive copies statements.json whatever
# its shape, and the server still reads one of these (`provenance.statements_record`).
_STATEMENTS = {
    "rows": [
        {"statement": {"type": "IncreaseAmount", "obj": {"name": "GliA"}}, "policy": "mass_action"}
    ],
    "mechanisms": {},
    "starts": {"GliA": 0.0},
    "readouts": {"obs_gli": "GliA"},
    "amounts": {},
}

#: What lands in an archived version: the plain files, and optimization.json gzipped.
_ARCHIVED = {
    "meta.json",
    "model.ant",
    "model_source.ant",
    "parameters.tsv",
    "pypesto_result.hdf5",
    "simulations.json",
    "statements.json",
    "validation_simulations.json",
    "optimization.json.gz",
}


def _work(tmp_path):
    """A work dir shaped like a finished run's: the version store plus both logs, and a version
    holding every file the server writes into one."""
    work = tmp_path / "work"
    store = work / ".petab_mcp"
    (store / "versions" / "v0").mkdir(parents=True)
    (store / "provenance.jsonl").write_text(
        json.dumps({"type": "edit", "version": 0}) + "\n", encoding="utf-8"
    )
    (store / "errors.jsonl").write_text(
        json.dumps(
            {
                "tool": "make_changes",
                "error_type": "ValidationError",
                "message": "unknown parameter id 'k_typo'",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (store / "HEAD").write_text("0", encoding="utf-8")
    v0 = store / "versions" / "v0"
    (v0 / "meta.json").write_text(json.dumps({"version": 0, "bic": 3.0}), encoding="utf-8")
    (v0 / "model.ant").write_text("model m\nend\n", encoding="utf-8")
    (v0 / "model_source.ant").write_text("model m\n  r1: A => B; k*A;\nend\n", encoding="utf-8")
    (v0 / "parameters.tsv").write_text("parameterId\testimate\nk1\t1\nsigma\t1\n", encoding="utf-8")
    (v0 / "simulations.json").write_text(json.dumps({"n_data_points": 6}), encoding="utf-8")
    (v0 / "validation_simulations.json").write_text(
        json.dumps({"n_data_points": 2}), encoding="utf-8"
    )
    (v0 / "statements.json").write_text(json.dumps(_STATEMENTS, indent=2), encoding="utf-8")
    (v0 / "optimization.json").write_text(json.dumps(_OPTIMIZATION, indent=2), encoding="utf-8")
    (v0 / "pypesto_result.hdf5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 32)
    # Deliberately NOT copied: the TSV duplicates simulations.json.
    (v0 / "simulations.tsv").write_text("observableId\tsimulation\n", encoding="utf-8")
    return work


def _archived_v0(tmp_path, run_id="t"):
    dest = archive_run(_work(tmp_path), {"phoenix_trace_id": run_id}, tmp_path / "archive")
    assert dest is not None
    return dest / ".petab_mcp" / "versions" / "v0"


def test_the_error_log_is_archived_with_the_event_log(tmp_path):
    """`provenance.jsonl` says what LANDED; `errors.jsonl` says what was REFUSED — tool,
    arguments and message per rejected call. Without the second, a finished run can be asked
    "what did it build?" but never "what did it try?", which is the question every editing A/B
    turns on (the 81%-statically-detectable figure that motivated draft editing, and the 0/120
    re-measurement that retired it)."""
    dest = archive_run(
        _work(tmp_path), {"phoenix_trace_id": "t-errors"}, archive_root=tmp_path / "archive"
    )
    assert dest is not None
    store = dest / ".petab_mcp"
    assert (store / "errors.jsonl").is_file(), "the error log did not survive archiving"
    rec = json.loads((store / "errors.jsonl").read_text().splitlines()[0])
    assert rec["message"] == "unknown parameter id 'k_typo'"  # content, not just the file
    assert (store / "provenance.jsonl").is_file()
    assert (store / "HEAD").read_text() == "0"


def test_a_run_with_no_errors_still_archives(tmp_path):
    """A clean run never creates `errors.jsonl` (it is opened on first append), so its ABSENCE
    must be ordinary — not a swallowed exception that silently drops the rest of the archive."""
    work = _work(tmp_path)
    (work / ".petab_mcp" / "errors.jsonl").unlink()
    dest = archive_run(work, {"phoenix_trace_id": "t-clean"}, archive_root=tmp_path / "archive")
    assert dest is not None
    store = dest / ".petab_mcp"
    assert not (store / "errors.jsonl").exists()
    assert (store / "provenance.jsonl").is_file()  # the rest still landed
    assert (store / "versions" / "v0" / "model.ant").is_file()


def test_the_version_source_is_archived(tmp_path):
    """`model_source.ant` is what makes lineage checkable after the fact (`child.source ==
    parent.source + ops`). Left out of _VERSION_FILES it dies with work/ on the next run of the
    same example — the identical silent loss errors.jsonl had, and nothing else would notice."""
    dest = archive_run(
        _work(tmp_path), {"phoenix_trace_id": "t-src"}, archive_root=tmp_path / "archive"
    )
    src = dest / ".petab_mcp" / "versions" / "v0" / "model_source.ant"
    assert src.is_file(), "the per-version source did not survive archiving"
    assert "r1: A => B; k*A;" in src.read_text()


def test_a_version_keeps_its_statements_multistart_and_hdf5(tmp_path):
    """The fontaine_sag INDRA sweep (#62) lost every version's `statements.json`, which was simply
    missing from the list, and its `optimization.json`, which was skipped on purpose; the sweep
    task then removed work/, so the model the agent wrote and the multistart behind each score
    survived nowhere on disk. Both are archived now, the multistart gzipped, beside pyPESTO's HDF5
    byte for byte. simulations.tsv (simulations.json's table again) stays out."""
    v0 = _archived_v0(tmp_path)
    assert {p.name for p in v0.iterdir()} == _ARCHIVED
    assert json.loads((v0 / "statements.json").read_text()) == _STATEMENTS
    with gzip.open(v0 / "optimization.json.gz", "rt", encoding="utf-8") as fh:
        assert json.load(fh) == _OPTIMIZATION
    live = tmp_path / "work" / ".petab_mcp" / "versions" / "v0" / "pypesto_result.hdf5"
    assert (v0 / "pypesto_result.hdf5").read_bytes() == live.read_bytes()


def test_a_version_without_held_out_simulations_still_archives(tmp_path):
    """`validation_simulations.json` comes with frohlich-lab/PEtab-MCP#119, so a version written by
    an older server has none. Its absence must be ordinary, not an exception that the courtesy
    `except` swallows along with the rest of the archive."""
    work = _work(tmp_path)
    (work / ".petab_mcp" / "versions" / "v0" / "validation_simulations.json").unlink()
    dest = archive_run(work, {"phoenix_trace_id": "t-old"}, archive_root=tmp_path / "archive")
    v0 = dest / ".petab_mcp" / "versions" / "v0"
    assert {p.name for p in v0.iterdir()} == _ARCHIVED - {"validation_simulations.json"}


def test_an_archived_run_loads_like_the_live_one(tmp_path):
    """Read back through the loaders, the archive must give the payload the live work dir gave:
    the gzipped optimization.json under its plain name, and the explorer's variants unchanged.
    The `starts` assertion is what keeps the comparison honest; two loaders that both failed to
    find the file would agree on None."""
    v0 = _archived_v0(tmp_path, "t-load")
    live_v0 = tmp_path / "work" / ".petab_mcp" / "versions" / "v0"
    assert not (v0 / "optimization.json").exists()

    assert read_version_json(v0, "optimization.json") == _OPTIMIZATION
    assert read_version_json(v0, "optimization.json") == read_version_json(
        live_v0, "optimization.json"
    )
    assert read_version_json(v0, "statements.json") == _STATEMENTS

    (live,) = B.load_variants(tmp_path / "work", n_data_points=None)
    (archived,) = B.load_variants(v0.parents[2], n_data_points=None)
    assert archived == live
    assert archived["starts"] == {
        "n_starts": 3,
        "n_at_best": 2,
        "exitflag_counts": {"ExitFlag.FTOL": 2, "ExitFlag.MAXTIME": 1},
    }
    assert archived["statements"]["rows"] == _STATEMENTS["rows"]


def test_a_damaged_gzip_reads_as_missing(tmp_path):
    """The explorer builds a run page from every version; one truncated `optimization.json.gz`
    must cost that version its multistart, not the page."""
    v0 = _archived_v0(tmp_path, "t-trunc")
    gz = v0 / "optimization.json.gz"
    gz.write_bytes(gz.read_bytes()[:40])
    assert read_version_json(v0, "optimization.json") is None
    (variant,) = B.load_variants(v0.parents[2], n_data_points=None)
    assert variant["starts"] is None and variant["statements"] == _STATEMENTS


def test_archiving_never_raises(tmp_path):
    """The contract the runner relies on: archiving is a courtesy, so a missing work dir
    returns None rather than killing a finished run at the last step."""
    assert (
        archive_run(
            tmp_path / "does_not_exist",
            {"phoenix_trace_id": "t"},
            archive_root=tmp_path / "archive",
        )
        is not None
    )
    assert archive_run(tmp_path / "w", {}, archive_root=tmp_path / "no" / "\0bad") is None
