"""What ``archive_run`` must carry out of a work dir before the next run wipes it.

The archive is the only durable copy of a finished run's on-disk half (the Phoenix trace holds
the conversation; ``work/`` is ``rmtree``d by the next run of the same example). These tests pin
WHICH files that is — in particular ``errors.jsonl``, the record of what the server REFUSED,
which every editing A/B's rejection-class analysis reads and which used to be reconstructible
only from live work dirs.
"""

from __future__ import annotations

import json

from gemot.explorer.archive import archive_run


def _work(tmp_path):
    """A work dir shaped like a finished run's: the version store plus both logs."""
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
    # Deliberately NOT copied — the archive stays lean (see _VERSION_FILES).
    (v0 / "pypesto_result.hdf5").write_bytes(b"\x00" * 32)
    return work


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


def test_the_archive_stays_lean(tmp_path):
    """The per-version big blobs are skipped on purpose: the archive is kept small enough to
    hold every run of every sweep."""
    dest = archive_run(
        _work(tmp_path), {"phoenix_trace_id": "t-lean"}, archive_root=tmp_path / "archive"
    )
    v0 = dest / ".petab_mcp" / "versions" / "v0"
    assert (v0 / "meta.json").is_file()
    assert not (v0 / "pypesto_result.hdf5").exists()


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
