"""Supervised, ISOLATED bio-knowledge MCP sessions for the blinded ``biocurator`` (issue #86).

The biocurator queries two HOSTED bio-knowledge MCP servers (``runner.BIO_MCP_URLS``:
BioContextAI + OmniPath). Each ``streamablehttp_client`` + ``ClientSession`` spawns background
anyio tasks (a ``post_writer``, an SSE reader, a receive loop). Before this module those sessions
were opened on the SAME ``AsyncExitStack`` as the essential petab session, so their task groups'
cancel scopes sat on the run's own task: a transport failure on a bio session — a
``ConnectTimeout`` at the TLS handshake, or a ``ClosedResourceError`` in the background
``post_writer`` under load — cancelled the whole run and it aborted with
``stop_reason: "aborted: ClosedResourceError"``, even when the petab session and the
model-building were fine (issue #86, a regression from #85).

The fix is structural: run each bio session in its OWN detached ``asyncio`` task (a "session
holder"). An ``asyncio.create_task`` task is a *sibling*, not an ancestor, of the run's task, so a
crash inside it can never deliver a cancellation to the run — the bio failure is contained. The
run uses the live session only via ``session.call_tool(...)`` from its own task, which is safe:
the dangerous anyio cancel scopes (the two task groups) are entered AND exited entirely inside the
holder task, while ``call_tool``/``list_tools``/``initialize`` only send/receive on anyio
memory-object streams (cross-task safe on one event loop). When a bio session dies, a later bio
``call_tool`` from the run's task raises an ordinary ``Exception`` (``ClosedResourceError`` /
``EndOfStream`` / ``TimeoutError``) — never a run-cancelling ``BaseExceptionGroup`` — which the
agent's per-tool circuit breaker already catches, degrading the biocurator to "dumb".

The module is deliberately self-contained and imports ``mcp`` LAZILY (only inside the default
opener), so its supervision/teardown logic can be unit-tested with a fake opener and no live
servers (see ``tests/test_bio_session_isolation.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any, AsyncIterator, Callable

# --- transport hardening (issue #86 fix #2) ------------------------------------------------
# A GENEROUS httpx connect timeout for the bio transport: the abort in #86 was a ConnectTimeout at
# the TLS handshake under load, so the tight httpx default is replaced with a roomier one.
# sse_read_timeout is separate — the in-session streaming read window (long, for tool responses);
# it deliberately does NOT govern the startup handshake, which the per-attempt deadline below
# bounds so a slow server can't stall run startup. A hung server is retried a couple of times.
_BIO_TRANSPORT_TIMEOUT_S = 10.0    # httpx connect/write/pool timeout for a bio session
_BIO_SSE_READ_TIMEOUT_S = 120.0    # in-session streaming read window (tool responses), not startup
# Per-ATTEMPT budget for a full connect+initialize. It EXCEEDS the httpx connect timeout above so a
# real handshake isn't cut short, and the ~10s of headroom over it covers a normal initialize();
# it deliberately caps the whole startup handshake well under sse_read_timeout so a slow/hung
# server can't stall startup (the long streaming window is for live tool calls, not this handshake).
_BIO_CONNECT_TIMEOUT_S = 20.0
_BIO_CONNECT_RETRIES = 2           # extra attempts after the first (up to 3 total per server)
_BIO_CONNECT_BACKOFF_S = 2.0       # wait between connect attempts
# Grace period at teardown for holder tasks to exit cleanly on the shutdown signal before any
# still-wedged one is cancelled.
_BIO_TEARDOWN_GRACE_S = 5.0


# A bio "opener": given a URL, an async context manager yielding an initialized MCP session.
BioOpener = Callable[[str], "contextlib.AbstractAsyncContextManager[Any]"]


def _summarize_exc(exc: BaseException) -> str:
    """A short one-liner for an exception, unwrapping an ``anyio`` TaskGroup ``BaseExceptionGroup``
    down to its first leaf.

    Intentionally a small LOCAL copy of ``runner._summarize_exc`` (not a shared import): this module
    must import WITHOUT ``mcp`` (its logic is unit-tested by loading it by file path with a fake
    opener, see ``tests/test_bio_session_isolation.py``), while ``runner`` hard-imports ``mcp`` at
    module top and a path-loaded module cannot resolve a relative import — so neither importing from
    ``runner`` nor a shared relative-import helper is viable here. The behaviour is pinned by
    ``test_summarize_exc_unwraps_exception_group`` so the copy can't silently drift.
    ``BaseExceptionGroup`` is a builtin on the project's 3.12 floor."""
    seen: BaseException = exc
    while isinstance(seen, BaseExceptionGroup) and seen.exceptions:
        seen = seen.exceptions[0]
    return f"{type(seen).__name__}: {seen}"[:200]


@contextlib.asynccontextmanager
async def _default_bio_opener(url: str) -> AsyncIterator[Any]:
    """Open one bio-knowledge MCP session (the real, mcp-backed opener).

    ``mcp`` is imported HERE, not at module top, so importing this module (and unit-testing its
    supervision logic with a fake opener) needs no ``mcp`` install. Uses the generous bio
    transport timeouts above so a handshake under load isn't cut short by the tight default.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    # ``sse_read_timeout`` is passed unconditionally (the petab session already uses it, so the
    # pinned mcp supports it). The connect ``timeout`` is NEWLY used here, so pass it ONLY if this
    # mcp version's client actually accepts it — otherwise a signature mismatch would TypeError on
    # EVERY connect and silently disable the biocurator (dumb) on every run. Absent the kwarg we
    # simply fall back to the client's default connect timeout (the hardening no-ops, not the run).
    kwargs: dict[str, Any] = {"sse_read_timeout": _BIO_SSE_READ_TIMEOUT_S}
    try:
        if "timeout" in inspect.signature(streamablehttp_client).parameters:
            kwargs["timeout"] = _BIO_TRANSPORT_TIMEOUT_S
    except (ValueError, TypeError):  # pragma: no cover - unsignaturable callable: skip the hint
        pass

    async with streamablehttp_client(url, **kwargs) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _bio_session_runner(
    url: str,
    holder: dict[str, Any],
    ready: "asyncio.Event",
    shutdown: "asyncio.Event",
    opener: BioOpener,
    log: Callable[[str], Any] = print,
) -> None:
    """Own ONE bio session for its whole lifetime, in this (detached) task — the isolation
    boundary. Opens the session, hands it out via ``holder`` + ``ready``, then parks on
    ``shutdown`` until the run tears down. ANY failure is contained here and never propagates:

    * a CONNECT-phase failure records the error in ``holder`` and sets ``ready`` so
      :func:`open_bio_sessions` skips this server;
    * a LIVE-session failure (the #86 background crash) is logged and swallowed — the run's own
      task is unaffected and the biocurator simply loses this session (its ``call_tool`` s then
      fail as ordinary tool errors and it degrades to "dumb").

    A genuine teardown ``CancelledError`` — and a ``KeyboardInterrupt``/``SystemExit`` — is
    re-raised (never swallowed): the clean cancel path completes and is retrieved by the teardown
    collection, and an interpreter-level exit is not masked by this task.
    """
    try:
        async with opener(url) as session:
            holder["session"] = session
            ready.set()               # hand the live session out to the supervisor
            await shutdown.wait()     # keep it open until the run tears down
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise                         # teardown cancel / interpreter exit — never swallowed
    except BaseException as exc:      # noqa: BLE001 — isolation boundary: contain transport failures
        if not ready.is_set():
            # Connect-phase failure: report it so the supervisor can skip/retry this URL.
            holder["error"] = exc
            ready.set()
        else:
            # Live-session failure AFTER hand-out (the #86 crash): the session is now dead; the
            # biocurator's bio tool calls will fail as ordinary tool errors and it degrades. This
            # MUST NOT propagate to the run's task — that is the whole point of the isolation.
            log(f"[bio] WARNING: bio session {url} failed after connect "
                f"({_summarize_exc(exc)}); the biocurator loses it for the rest of the run.")
        # swallow: contain the failure in this detached task


@contextlib.asynccontextmanager
async def open_bio_sessions(
    urls: "tuple[str, ...] | list[str]",
    *,
    opener: BioOpener = _default_bio_opener,
    connect_timeout: float = _BIO_CONNECT_TIMEOUT_S,
    retries: int = _BIO_CONNECT_RETRIES,
    backoff: float = _BIO_CONNECT_BACKOFF_S,
    teardown_grace: float = _BIO_TEARDOWN_GRACE_S,
    log: Callable[[str], Any] = print,
) -> "AsyncIterator[list[Any]]":
    """Yield a list of LIVE, ISOLATED bio-knowledge MCP sessions (possibly empty).

    Each URL is opened in its own supervised :func:`_bio_session_runner` task, so a bio-session
    transport failure — at connect OR at any later point — degrades the biocurator instead of
    aborting the run (issue #86). Connect is best-effort with a per-attempt deadline and a couple
    of retries; an unreachable/slow server is warned and skipped. The yielded sessions stay live
    for the ``async with`` body and are torn down on exit.

    ``opener`` is injectable so the supervision/teardown logic can be unit-tested without ``mcp``.
    This context manager NEVER raises for a bio failure (its ``__aexit__`` must not convert a
    successful run into a degraded one).
    """
    # Imported HERE, not at module scope. This module is deliberately self-contained so its
    # supervision/teardown logic can be unit-tested by loading it BY FILE PATH with a fake opener
    # (see `_summarize_exc`, whose duplication exists for exactly that reason) — and a path-loaded
    # module cannot resolve a relative import. A top-level `from . import bio_cache` would have
    # silently removed that escape hatch and falsified the comment justifying the duplication.
    from . import bio_cache, omnipath_local

    shutdown = asyncio.Event()
    tasks: list[asyncio.Task] = []      # every holder task ever created — all retrieved at teardown
    sessions: list[Any] = []
    caches: list[Any] = []
    try:
        for url in urls:
            # OmniPath is served from a PINNED LOCAL SNAPSHOT when one is on disk: no network, no
            # rate limit, and — the part that matters for an A/B — every arm sees identical biology
            # instead of whatever the hosted service returns that day. A missing snapshot falls
            # straight through to the hosted service, so this can only ever degrade to the old
            # behaviour. See `omnipath_local` for why a cache could never have covered this endpoint
            # (free-text SQL: unbounded key space, measured 0.00 hit rate over 11 runs).
            if "omnipathdb" in url and omnipath_local.available():
                sessions.append(omnipath_local.LocalOmniPathSession())
                info = omnipath_local.snapshot_info()
                tables = info.get("tables") or {}
                counts = ", ".join(f"{name} {meta.get('rows')}" for name, meta in tables.items())
                log(f"[bio] OmniPath served LOCALLY from the {info.get('fetched_at')} snapshot "
                    f"({counts} rows) — no network, no rate limit")
                continue
            session = None
            for attempt in range(retries + 1):
                holder: dict[str, Any] = {"session": None, "error": None}
                ready = asyncio.Event()
                task = asyncio.create_task(
                    _bio_session_runner(url, holder, ready, shutdown, opener, log)
                )
                tasks.append(task)
                try:
                    await asyncio.wait_for(ready.wait(), timeout=connect_timeout)
                    timed_out = False
                except asyncio.TimeoutError:
                    timed_out = True

                if holder["session"] is not None:      # connected (even if right at the deadline)
                    session = holder["session"]
                    log(f"[bio] biocurator knowledge server connected: {url}")
                    break

                # This attempt failed: a hung connect (cancel it — it's kept for teardown) or a
                # connect error (the runner already finished, swallowed).
                if timed_out and not task.done():
                    task.cancel()
                if timed_out:
                    reason = f"connect exceeded {connect_timeout:g}s"
                elif holder["error"] is not None:
                    reason = _summarize_exc(holder["error"])
                else:
                    reason = "unknown error"
                if attempt < retries:
                    log(f"[bio] connect to bio-knowledge server {url} failed ({reason}); "
                        f"retry {attempt + 1}/{retries} in {backoff:g}s")
                    await asyncio.sleep(backoff)
                else:
                    log(f"[bio] WARNING: could not connect to bio-knowledge server {url} "
                        f"({reason}); skipping it.")
            if session is not None:
                # Read-through disk cache in front of the hosted server, wrapped HERE so every
                # consumer gets it and none has to know. A sweep asks the same SCENARIO-level
                # questions on every seed (each fujita run looks up the same four entities), so
                # after one warm run the rest are served from disk and never leave the cluster —
                # which is what stops the fair-use throttle being the limit on seed count, and
                # therefore on whether a sweep has the power to measure anything.
                cache = bio_cache.wrap_session(session, url, log=log)
                if cache.enabled:
                    caches.append(cache)
                sessions.append(session)
        if caches:
            log(f"[bio] response cache: {bio_cache.cache_dir()}")
        try:
            yield sessions
        finally:
            # Report the HIT RATE, not just the path. Without this a sweep logs
            # "[bio] response cache: <path>" identically whether the cache is serving everything or
            # nothing -- and it was serving nothing: a blanket kwargs bypass meant a 0% hit rate in
            # production that no log line and no test could reveal. A number here is the thing that
            # would have shown it on the first run.
            for c in caches:
                with contextlib.suppress(Exception):
                    s = c.stats()
                    log(f"[bio] cache {s['url']}: {s['hits']} hits / {s['misses']} misses "
                        f"(rate {s['hit_rate']}), {s['writes']} written, "
                        f"{s['bypasses']} bypassed, {s['errors']} errors")
    finally:
        # Signal the holders to close cleanly, give them a moment, then cancel any straggler and
        # RETRIEVE every task so nothing leaks an unretrieved exception AND this teardown never
        # raises a bio failure into the run's scope. The final collection is itself BOUNDED
        # (wait_for) so a holder that ignores cancellation — e.g. a wedged mcp session close, the
        # very failure mode this module guards against — cannot hang the run's teardown; a still-
        # wedged task is left for the event loop to reap at close.
        shutdown.set()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=teardown_grace)
            for task in pending:
                task.cancel()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=teardown_grace)
