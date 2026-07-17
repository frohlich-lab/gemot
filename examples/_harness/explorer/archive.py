"""Snapshot a run's on-disk half at run end, before the next ``cleanup()`` wipes ``work/``.

The Phoenix trace (conversation/tokens/sub-agents) persists in the collector on its own and is
fetched lazily by the viewer via ``phoenix_trace_id``. What does NOT survive is the server's
version store under ``work/.petab_mcp`` — the next run of the same example ``rmtree``s ``work/``.
So at run end we copy the lightweight, durable pieces (``result.json`` + each version's
``meta.json`` / ``model.ant`` / ``parameters.tsv`` / ``simulations.json`` + ``provenance.jsonl``)
into ``examples/_runs_archive/<run_id>/``, laid out as ``.petab_mcp/`` so :mod:`.bundle` reads it
unchanged. Best-effort: any failure is swallowed so archiving never breaks a run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# per-version files worth keeping — small + sufficient for the trajectory panel. The big
# pypesto_result.hdf5 / optimization.json are deliberately skipped to keep the archive lean.
_VERSION_FILES = ("meta.json", "model.ant", "parameters.tsv", "simulations.json")


def default_archive_root() -> Path:
    # examples/_harness/explorer/archive.py -> examples/_runs_archive
    return Path(__file__).resolve().parents[2] / "_runs_archive"


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
        for name in ("provenance.jsonl", "HEAD"):
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
        runs.append({
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
        })
    runs.sort(key=lambda x: (x.get("run_started_at") or ""), reverse=True)
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
