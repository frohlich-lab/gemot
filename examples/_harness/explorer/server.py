"""Standalone FastAPI app serving the run explorer.

Not coupled to the MCP server (which is single-problem, stdio-first). Reads archived runs from
``examples/_runs_archive`` and fetches each run's Phoenix trace lazily (cached) when its page is
opened. Build via :func:`create_app`; launch via ``examples/explore_ui.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from . import archive, bundle

_HERE = Path(__file__).resolve().parent


def _phoenix_base_url(explicit: str | None) -> str:
    url = (explicit or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
           or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "http://localhost:6006")
    return url.rstrip("/")


def create_app(archive_root: Path | None = None, phoenix_base_url: str | None = None) -> FastAPI:
    archive_root = Path(archive_root) if archive_root else archive.default_archive_root()
    px_url = _phoenix_base_url(phoenix_base_url)
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    cache: dict[str, dict] = {}

    app = FastAPI(title="petab-mcp run explorer")
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html",
            {"runs": archive.list_runs(archive_root),
             "archive_root": str(archive_root), "phoenix": px_url},
        )

    @app.get("/run/{run_id}")
    def run_page(request: Request, run_id: str):
        if archive.load_run_meta(run_id, archive_root) is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return templates.TemplateResponse(request, "run.html", {"run_id": run_id})

    @app.get("/api/run/{run_id}")
    def run_bundle(run_id: str, refresh: bool = False):
        if refresh:
            cache.pop(run_id, None)
        if run_id in cache:
            return JSONResponse(cache[run_id])
        loaded = archive.load_run_meta(run_id, archive_root)
        if loaded is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        run_dir, meta = loaded
        data = bundle.build_bundle(run_id, run_dir=run_dir, meta=meta,
                                   phoenix_base_url=px_url, project=meta.get("phoenix_project"))
        # cache only once the Phoenix half succeeded (so a not-yet-ingested trace can be retried)
        if data.get("phoenix", {}).get("ok") or not meta.get("phoenix_trace_id"):
            cache[run_id] = data
        return JSONResponse(data)

    return app
