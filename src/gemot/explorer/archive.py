"""Snapshot a run's on-disk half at run end, before the next ``cleanup()`` wipes ``work/``.

The Phoenix trace (conversation/tokens/sub-agents) persists in the collector on its own and is
fetched lazily by the viewer via ``phoenix_trace_id``. What does NOT survive is the server's
version store under ``work/.petab_mcp`` — the next run of the same example ``rmtree``s ``work/``,
and a sweep task removes its own when it exits. So at run end we copy the durable pieces
(``result.json``, ``provenance.jsonl``, ``errors.jsonl`` and, per version,
:data:`_VERSION_FILES` plus a gzipped ``optimization.json``) into
``examples/_runs_archive/<run_id>/``, laid out as ``.petab_mcp/`` so :mod:`.bundle` reads it
unchanged; :func:`gemot.analysis.read_version_json` reads a gzipped file under its plain name.
Best-effort: any failure is swallowed so archiving never breaks a run.

``simulations.tsv`` is never archived: ``simulations.json`` holds the same table and is the one
every reader reads.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from .. import paths

#: Per-version files copied as they are. A file a version lacks is skipped, which is ordinary:
#: only a model built from statements has ``statements.json``, and only a server with
#: frohlich-lab/PEtab-MCP#119 writes ``validation_simulations.json``.
_VERSION_FILES = (
    "meta.json",
    "model.ant",
    "parameters.tsv",
    "simulations.json",
    # The caller's own text. Without it the post-hoc lineage check the source
    # exists for (`child.source == parent.source + ops`) dies with work/ on the
    # next run of the same example — the identical hole errors.jsonl had.
    "model_source.ant",
    # The INDRA arm's model as the agent wrote it: rows, mechanisms, starts, amounts, readouts.
    # model.ant is only its BioNetGen expansion (species `__s0`, reactions `r0`...), so without
    # this the statements of a finished run survived only in its Phoenix trace, and could be
    # neither rendered, nor resubmitted, nor counted (#62, the fontaine_sag sweep). 2-17 kB.
    "statements.json",
    # The held-out simulations, beside simulations.json (frohlich-lab/PEtab-MCP#119).
    "validation_simulations.json",
    # pyPESTO's own result for the fit, readable by pyPESTO itself. Already compressed, so it is
    # copied as it is, at up to ~5 MB a version.
    "pypesto_result.hdf5",
)

#: Per-version files archived gzipped, as ``<name>.gz``. ``optimization.json`` is the multistart
#: behind a version's score — every start's ``x``, ``fval`` and time, the exit-flag histogram, the
#: per-iteration trace — and without it a finished run cannot say whether its best was found by
#: one start or by many, nor which parameters sit at bounds across starts (#62). 1.4-1.9 MB raw
#: for a 50-start fit, about a fifth of that gzipped.
_GZIPPED_VERSION_FILES = ("optimization.json",)


def default_archive_root() -> Path:
    return paths.runs_archive_root()


def default_examples_root() -> Path:
    # holds each example's live work/
    return paths.examples_root()


# Marker file the runner writes into ``work/`` while a run is in flight (see
# runner._write_live_marker). Its presence WITHOUT a sibling ``result.json`` means "running".
LIVE_MARKER = "run_live.json"


def _live_markers(examples_root: Path) -> list[Path]:
    """Paths of live markers for runs still in flight (``examples/<ex>/work/run_live.json``
    with no sibling ``result.json`` yet). A finished run has a ``result.json`` and is served
    from the archive instead, so it drops out of the live set here."""
    if not examples_root.is_dir():
        return []
    out = []
    for mp in sorted(examples_root.glob(f"*/work/{LIVE_MARKER}")):
        if not (mp.parent / "result.json").is_file():
            out.append(mp)
    return out


def list_live_runs(examples_root: Path | None = None) -> list[dict]:
    """Summarize in-progress runs (newest first) for the run-list page's live section."""
    root = examples_root or default_examples_root()
    runs = []
    for mp in _live_markers(root):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        runs.append(
            {
                "run_id": m.get("run_id") or m.get("phoenix_trace_id"),
                "scenario": m.get("scenario"),
                "model": m.get("model"),
                "driver": m.get("driver"),
                "run_started_at": m.get("run_started_at"),
                "has_trace": bool(m.get("phoenix_trace_id")),
                "n_versions": _count_versions(mp.parent),  # mp.parent is the live work/ dir
                "live": True,
            }
        )
    runs.sort(key=lambda x: x.get("run_started_at") or "", reverse=True)
    return runs


def load_live_run(run_id: str, examples_root: Path | None = None) -> tuple[Path, dict] | None:
    """Return ``(work_dir, marker_meta)`` for an in-progress run, or None if it isn't live
    (never started, or already finished → served from the archive)."""
    root = examples_root or default_examples_root()
    for mp in _live_markers(root):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        if (m.get("run_id") or m.get("phoenix_trace_id")) == run_id:
            return mp.parent, m
    return None


def run_id_for(result: dict) -> str:
    tid = result.get("phoenix_trace_id")
    if tid:
        return str(tid)
    stamp = str(result.get("run_started_at") or "").replace(":", "").replace(".", "")
    return f"scripted-{stamp or 'unknown'}"


def archive_run(work: Path, result: dict, archive_root: Path | None = None) -> Path | None:
    """Copy the run's durable on-disk artifacts into the archive. Returns the dest dir or None.

    Never raises — archiving is a courtesy, not part of the run's contract.
    """
    try:
        work = Path(work)
        root = archive_root or default_archive_root()
        dest = root / run_id_for(result)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(json.dumps(result, indent=2))

        store = work / ".petab_mcp"
        dest_store = dest / ".petab_mcp"
        # `errors.jsonl` rides along with the event log because it is the only durable record of
        # what the agent got WRONG: every rejected call, with the tool, the arguments and the
        # message (petab_mcp.provenance.log_error). provenance.jsonl records what LANDED, so
        # without this a finished run can be asked "what did it build?" but never "what did it
        # try that the server refused?" — which is the rejection-class question every editing
        # A/B has turned on (the 81%-statically-detectable figure that motivated draft editing,
        # and the 0/120 re-measurement that retired it, both had to be reconstructed from live
        # work dirs that the next run then wiped). It is small: 2-8 kB against provenance's
        # 37-46 kB on the sweep runs, because it holds only failures.
        for name in ("provenance.jsonl", "errors.jsonl", "HEAD"):
            src = store / name
            if src.is_file():
                dest_store.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_store / name)
        vsrc = store / "versions"
        if vsrc.is_dir():
            for vd in vsrc.glob("v*"):
                out = dest_store / "versions" / vd.name
                out.mkdir(parents=True, exist_ok=True)
                for fn in _VERSION_FILES:
                    f = vd / fn
                    if f.is_file():
                        shutil.copy2(f, out / fn)
                for fn in _GZIPPED_VERSION_FILES:
                    f = vd / fn
                    if f.is_file():
                        with f.open("rb") as src, gzip.open(out / f"{fn}.gz", "wb") as gz:
                            shutil.copyfileobj(src, gz)
        return dest
    except Exception:
        return None


def list_runs(archive_root: Path | None = None) -> list[dict]:
    """Summarize archived runs (newest first) for the run-list page."""
    root = archive_root or default_archive_root()
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        rj = d / "result.json"
        if not rj.is_file():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        runs.append(
            {
                "run_id": d.name,
                "scenario": r.get("scenario"),
                "model": r.get("model"),
                "driver": r.get("driver"),
                "success": r.get("success"),
                "stop_reason": r.get("stop_reason"),
                "delta_bic": r.get("delta_bic"),
                "bic": r.get("bic"),
                "run_started_at": r.get("run_started_at"),
                "run_ended_at": r.get("run_ended_at"),
                "has_trace": bool(r.get("phoenix_trace_id")),
                "n_versions": _count_versions(d),
            }
        )
    runs.sort(key=lambda x: x.get("run_started_at") or "", reverse=True)
    return runs


def load_run_meta(run_id: str, archive_root: Path | None = None) -> tuple[Path, dict] | None:
    root = archive_root or default_archive_root()
    d = root / run_id
    rj = d / "result.json"
    if not rj.is_file():
        return None
    try:
        return d, json.loads(rj.read_text())
    except Exception:
        return None


def _count_versions(d: Path) -> int:
    v = d / ".petab_mcp" / "versions"
    return len(list(v.glob("v*"))) if v.is_dir() else 0
