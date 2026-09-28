"""Standalone FastAPI app serving the run explorer.

Not coupled to the MCP server (which is single-problem, stdio-first). Reads archived runs from
``examples/_runs_archive`` and fetches each run's Phoenix trace lazily (cached) when its page is
opened. Build via :func:`create_app`; launch via ``examples/explore_ui.py``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from . import archive, bundle

_HERE = Path(__file__).resolve().parent

# Cap on the archived-bundle cache (see create_app). The explorer is a long-lived service (up to
# EXPLORER_TIME, ~1 day), so an unbounded dict would grow by one full run bundle for every
# distinct archived run ever opened; a bounded LRU keeps the hot set cached without that leak.
_ARCHIVED_CACHE_MAX = 256


def _phoenix_base_url(explicit: str | None) -> str:
    # Resolve the endpoint (explicit arg, then the standard env vars, then the local default),
    # then normalize to the base via the shared helper so a `.../v1/traces` env value is stripped
    # to the base the Phoenix Client wants — same rule as metrics/experiments/tracing.
    from ..tracing import phoenix_base_url

    url = (
        explicit
        or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "http://localhost:6006"
    )
    return phoenix_base_url(url)


def create_app(
    archive_root: Path | None = None,
    phoenix_base_url: str | None = None,
    examples_root: Path | None = None,
) -> FastAPI:
    archive_root = Path(archive_root) if archive_root else archive.default_archive_root()
    # Live markers live in each example's work/ dir; that tree is the archive's parent
    # (examples/), so derive it from archive_root unless overridden.
    examples_root = Path(examples_root) if examples_root else archive_root.parent
    px_url = _phoenix_base_url(phoenix_base_url)
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    cache: "OrderedDict[str, dict]" = OrderedDict()  # archived bundles, LRU-bounded (see below)
    live_cache: dict[str, tuple[float, dict]] = {}  # run_id -> (monotonic_ts, bundle)
    # Seconds a live bundle is reused before re-fetching. Kept fairly high because each refresh
    # re-queries the whole Phoenix trace (an Arrow->pandas build that is the process's heaviest,
    # and least stable, operation); a run only commits a model every ~2 min, so 15s is plenty
    # fresh while keeping the trace-fetch rate — and the native-crash risk — low.
    live_ttl = 15.0
    # Serialize every Phoenix trace fetch: uvicorn dispatches these sync handlers across a
    # threadpool, and the live poll loop can have several in flight at once. The underlying
    # pandas/pyarrow trace-dataframe machinery is not safe under concurrent access and will
    # segfault the process — this lock (held for the whole bundle build) prevents that.
    fetch_lock = threading.Lock()
    # Guards the archived-bundle `cache` (the read/insert/LRU-evict below): uvicorn dispatches
    # these sync handlers across a threadpool, so concurrent polls could otherwise evict a key
    # between another request's membership check and its read — a KeyError. Held only for the
    # cheap dict ops, never around the heavy _build().
    cache_lock = threading.Lock()

    def _build(run_id: str, run_dir, meta: dict) -> dict:
        kw = dict(
            run_dir=run_dir, meta=meta, phoenix_base_url=px_url, project=meta.get("phoenix_project")
        )
        # Only the Phoenix trace fetch (Arrow/pandas) is thread-unsafe, so take the lock solely
        # for runs that have a trace — trace-less bundles don't serialize under concurrent users.
        if meta.get("phoenix_trace_id"):
            with fetch_lock:
                return bundle.build_bundle(run_id, **kw)
        return bundle.build_bundle(run_id, **kw)

    app = FastAPI(title="petab-mcp run explorer")
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "runs": archive.list_runs(archive_root),
                "live_runs": archive.list_live_runs(examples_root),
                "archive_root": str(archive_root),
                "phoenix": px_url,
            },
        )

    @app.get("/api/live")
    def live_list():
        """The in-progress runs — polled by the index so it refreshes when a run starts or
        finishes (without the run pages having to reload)."""
        return JSONResponse(archive.list_live_runs(examples_root))

    @app.get("/run/{run_id}")
    def run_page(request: Request, run_id: str):
        if (
            archive.load_run_meta(run_id, archive_root) is None
            and archive.load_live_run(run_id, examples_root) is None
        ):
            raise HTTPException(404, f"unknown run {run_id!r}")
        return templates.TemplateResponse(request, "run.html", {"run_id": run_id})

    @app.get("/api/run/{run_id}")
    def run_bundle(run_id: str, refresh: bool = False):
        # A live (in-flight) run is served from its work/ dir with only a short TTL cache — it
        # is changing under us, so each poll re-reads the version store and re-queries Phoenix,
        # but a burst of near-simultaneous polls (several tabs) reuses one fetch within the TTL.
        live = archive.load_live_run(run_id, examples_root)
        if live is not None:
            hit = live_cache.get(run_id)
            if hit is not None and (time.monotonic() - hit[0]) < live_ttl:
                return JSONResponse(hit[1])
            work_dir, meta = live
            data = _build(run_id, work_dir, meta)
            data["live"] = True
            live_cache[run_id] = (time.monotonic(), data)
            return JSONResponse(data)
        live_cache.pop(run_id, None)  # finished — let the archived path take over

        # Otherwise the run is finished (or never was live): serve the archived bundle. Once a
        # run the client was watching finishes, this path takes over and returns live=false,
        # which stops the client polling — a seamless hand-off from the live view.
        with cache_lock:
            if refresh:
                cache.pop(run_id, None)
            cached = cache.get(run_id)
            if cached is not None:
                cache.move_to_end(run_id)  # LRU: mark as most-recently used
        if cached is not None:
            return JSONResponse(cached)
        loaded = archive.load_run_meta(run_id, archive_root)
        if loaded is None:
            raise HTTPException(404, f"unknown run {run_id!r}")
        run_dir, meta = loaded
        data = _build(run_id, run_dir, meta)  # heavy; outside cache_lock (uses its own fetch_lock)
        data["live"] = False
        # cache only once the Phoenix half succeeded (so a not-yet-ingested trace can be retried)
        if data.get("phoenix", {}).get("ok") or not meta.get("phoenix_trace_id"):
            with cache_lock:
                cache[run_id] = data
                cache.move_to_end(run_id)
                while len(cache) > _ARCHIVED_CACHE_MAX:  # evict least-recently-used bundles
                    cache.popitem(last=False)
        return JSONResponse(data)

    return app
