"""Supervised, ISOLATED bio-knowledge MCP sessions for the blinded ``biocurator`` (issue #86).

The biocurator queries two HOSTED bio-knowledge MCP servers (``runner.BIO_MCP_URLS``:
BioContextAI + OmniPath). Each ``streamable_http_client`` + ``ClientSession`` spawns background
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
``call_tool`` from the run's task raises an ordinary ``Exception`` — never a run-cancelling
``BaseExceptionGroup`` — which the agent's per-tool circuit breaker catches, cutting that tool.

A session that dies mid-run is REOPENED. Under mcp 2 a hosted server gone away does not fail the
call that met it: the transport's httpx2 error ends the holder task, and every call on the session
from then on raises ``MCPError("Connection closed")``, even once the server is back. Nothing used
to open a new one, so the breaker's cut held for the rest of the run: four tools for 26 h in one
run of campaign gemot_pools_20261107 (frohlich-lab/PEtab-MCP#337, finding C25). So the run holds a
:class:`BioSession` per server, not the mcp session: a handle whose live session a keeper task
(:func:`_keep_open`, detached like the holders) replaces, with backoff, each time it closes. What
was built on the handle — the tools, and the cache shadowing its ``call_tool``
(``bio_cache.wrap_session``) — outlives any one session, so neither is rebuilt. A call that finds
the session closed raises :class:`BioSessionClosed`, which the breaker cuts the tool on and gives
it back for once the session is open again (:meth:`BioSession.when_open`).

The module is deliberately self-contained and imports ``mcp`` LAZILY (only inside the default
opener), so its supervision/teardown logic can be unit-tested with a fake opener and no live
servers (see ``tests/test_bio_session_isolation.py`` and ``tests/test_bio_session_reopen.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator, Callable

# --- transport hardening (issue #86 fix #2) ------------------------------------------------
# A GENEROUS httpx connect timeout for the bio transport: the abort in #86 was a ConnectTimeout at
# the TLS handshake under load, so the tight httpx default is replaced with a roomier one.
# The read timeout is separate — the in-session streaming read window (long, for tool responses);
# it deliberately does NOT govern the startup handshake, which the per-attempt deadline below
# bounds so a slow server can't stall run startup. A hung server is retried a couple of times.
_BIO_TRANSPORT_TIMEOUT_S = 10.0  # httpx connect/write/pool timeout for a bio session
_BIO_SSE_READ_TIMEOUT_S = 120.0  # in-session streaming read window (tool responses), not startup
# Per-ATTEMPT budget for a full connect+initialize. It EXCEEDS the httpx connect timeout above so a
# real handshake isn't cut short, and the ~10s of headroom over it covers a normal initialize();
# it deliberately caps the whole startup handshake well under the read timeout so a slow/hung
# server can't stall startup (the long streaming window is for live tool calls, not this handshake).
_BIO_CONNECT_TIMEOUT_S = 20.0
_BIO_CONNECT_RETRIES = 2  # extra attempts after the first (up to 3 total per server)
_BIO_CONNECT_BACKOFF_S = 2.0  # wait between connect attempts
#: Cap on the wait between attempts to reopen a session that closed mid-run. The wait starts at
#: ``_BIO_CONNECT_BACKOFF_S`` and doubles. A server that never comes back then costs one connect
#: attempt every five minutes, which a fair-use endpoint does not notice, and one that does is
#: back in a run measured in hours within minutes of its server.
_BIO_REOPEN_MAX_BACKOFF_S = 300.0
# Grace period at teardown for holder tasks to exit cleanly on the shutdown signal before any
# still-wedged one is cancelled.
_BIO_TEARDOWN_GRACE_S = 5.0
#: The largest server-sent event a bio session reads, 16 MiB where mcp's default is 1 MiB. Still a
#: bound, since the hosted servers are not the run's own, but one a whole UniProt entry fits in
#: (TP53's reply is about 2 MB; see :mod:`gemot.agent.uniprot`).
_BIO_MAX_SSE_EVENT_BYTES = 16 * 1024 * 1024


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


class BioSessionClosed(ConnectionError):
    """A call on a hosted bio server whose session has closed, and is being reopened.

    A ``ConnectionError``, so the circuit breaker cuts the tool as it does on any transport error;
    ``session`` is the :class:`BioSession` it hands the tool back through."""

    def __init__(self, session: BioSession, cause: str) -> None:
        super().__init__(f"the session to {session.url} closed ({cause}) and is being reopened")
        self.session = session


class BioSession:
    """One hosted bio server as the run holds it, whichever mcp session is live behind it.

    Only ``call_tool`` and ``list_tools`` reach that session: they are all the tool bridge
    (``agent.mcp_tools``) and the cache call. One that fails on a transport error drops the session,
    wakes the keeper (:func:`_keep_open`) to reopen it, and raises :class:`BioSessionClosed`; so
    does a call made while it is being reopened. A transport error, not only the ``MCPError`` a
    closed session raises under mcp 2, because it is the same test the breaker cuts the tool on: a
    session the run cannot use is no use to keep.
    """

    def __init__(self, url: str, log: Callable[[str], Any] = print) -> None:
        self.url = url
        #: The live mcp session, or ``None`` while it is being reopened.
        self.session: Any = None
        #: Sessions opened so far: 1 for one that never closed.
        self.opened = 0
        self._log = log
        self._lost = asyncio.Event()  # the live session's: set when a call finds it unusable
        self._waiting: list[Callable[[], Any]] = []

    async def call_tool(self, name: str, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await self._through("call_tool", name, arguments, *args, **kwargs)

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        return await self._through("list_tools", *args, **kwargs)

    async def _through(self, method: str, *args: Any, **kwargs: Any) -> Any:
        session = self.session
        if session is None:
            raise BioSessionClosed(self, "not reopened yet")
        try:
            return await getattr(session, method)(*args, **kwargs)
        except Exception as exc:
            # The breaker's own test, so the two agree on what a transport error is. Imported
            # here because the breaker imports this module.
            from .agent.circuit_breaker import _is_transport_error

            if not _is_transport_error(exc):
                raise
            self._drop(session)
            raise BioSessionClosed(self, _summarize_exc(exc)) from exc

    def when_open(self, callback: Callable[[], Any]) -> None:
        """Call ``callback`` once a session is open: now if one is, else when it has been reopened."""
        if self.session is not None:
            callback()
        else:
            self._waiting.append(callback)

    def _open(self, session: Any, lost: asyncio.Event) -> None:
        """Make ``session`` the live one; ``lost`` is the event its holder parks on."""
        self.session, self._lost = session, lost
        self.opened += 1
        waiting, self._waiting = self._waiting, []
        for callback in waiting:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - one callback must not stop the others
                self._log(f"[bio] WARNING: a callback on reopening {self.url} failed: {exc!r}")

    def _drop(self, session: Any) -> None:
        """``session`` is unusable: if it is still the live one, drop it and wake its holder."""
        if session is not None and session is self.session:
            self.session = None
            self._lost.set()


async def _until_set(*events: asyncio.Event) -> None:
    """Return once any of ``events`` is set."""
    waits = [asyncio.ensure_future(event.wait()) for event in events]
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()


@contextlib.asynccontextmanager
async def _default_bio_opener(url: str) -> AsyncIterator[Any]:
    """Open one bio-knowledge MCP session (the real, mcp-backed opener).

    ``mcp`` is imported HERE, not at module top, so importing this module (and unit-testing its
    supervision logic with a fake opener) needs no ``mcp`` install. Uses the generous bio
    transport timeouts above so a handshake under load isn't cut short by the tight default.
    """
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # Both timeouts live on the httpx2 client the transport runs on (mcp 1.x took them as the
    # transport's `timeout` and `sse_read_timeout`).
    timeout = httpx2.Timeout(_BIO_TRANSPORT_TIMEOUT_S, read=_BIO_SSE_READ_TIMEOUT_S)
    async with httpx2.AsyncClient(timeout=timeout) as http:
        async with streamable_http_client(
            url, http_client=http, max_sse_event_size=_BIO_MAX_SSE_EVENT_BYTES
        ) as (read, write):
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
    *,
    lost: "asyncio.Event | None" = None,
) -> None:
    """Own ONE bio session for its whole lifetime, in this (detached) task — the isolation
    boundary. Opens the session, hands it out via ``holder`` + ``ready``, then parks on
    ``shutdown`` until the run tears down, or on ``lost`` until a call finds the session unusable
    (:meth:`BioSession._drop`), and closes it. ANY failure is contained here and never propagates:

    * a CONNECT-phase failure records the error in ``holder`` and sets ``ready`` so
      :func:`open_bio_sessions` skips this server;
    * a LIVE-session failure (the #86 background crash) is logged and swallowed — the run's own
      task is unaffected, and the session's keeper (:func:`_keep_open`) sees this task end and
      reopens it.

    A genuine teardown ``CancelledError`` — and a ``KeyboardInterrupt``/``SystemExit`` — is
    re-raised (never swallowed): the clean cancel path completes and is retrieved by the teardown
    collection, and an interpreter-level exit is not masked by this task.
    """
    try:
        async with opener(url) as session:
            holder["session"] = session
            ready.set()  # hand the live session out to the supervisor
            # Keep it open until the run tears down, or until a call finds it unusable.
            await _until_set(shutdown, *([lost] if lost is not None else []))
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise  # teardown cancel / interpreter exit — never swallowed
    except BaseException as exc:  # noqa: BLE001 — isolation boundary: contain transport failures
        if not ready.is_set():
            # Connect-phase failure: report it so the supervisor can skip/retry this URL.
            holder["error"] = exc
            ready.set()
        else:
            # Live-session failure AFTER hand-out (the #86 crash): the session is now dead. This
            # MUST NOT propagate to the run's task — that is the whole point of the isolation.
            log(f"[bio] WARNING: bio session {url} failed after connect ({_summarize_exc(exc)}).")
        # swallow: contain the failure in this detached task


async def _connect(
    url: str,
    *,
    opener: BioOpener,
    shutdown: asyncio.Event,
    lost: asyncio.Event,
    tasks: list[asyncio.Task],
    connect_timeout: float,
    retries: int,
    backoff: float,
    log: Callable[[str], Any],
) -> "tuple[Any, asyncio.Task | None, str]":
    """Open ``url``'s session in a holder task of its own: ``(session, holder, "")``, or
    ``(None, None, why)`` once ``retries`` more attempts have failed too.

    Each attempt is bounded by ``connect_timeout`` and one that hangs is cancelled. Every holder
    goes into ``tasks`` for the teardown to collect; ``lost`` is what the live one parks on."""
    reason = ""
    for attempt in range(retries + 1):
        holder: dict[str, Any] = {"session": None, "error": None}
        ready = asyncio.Event()
        task = asyncio.create_task(
            _bio_session_runner(url, holder, ready, shutdown, opener, log, lost=lost)
        )
        tasks.append(task)
        try:
            await asyncio.wait_for(ready.wait(), timeout=connect_timeout)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True

        if holder["session"] is not None:  # connected (even if right at the deadline)
            return holder["session"], task, ""

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
            log(
                f"[bio] connect to bio-knowledge server {url} failed ({reason}); "
                f"retry {attempt + 1}/{retries} in {backoff:g}s"
            )
            await asyncio.sleep(backoff)
    return None, None, reason


async def _keep_open(
    bio: BioSession,
    holder: asyncio.Task,
    *,
    opener: BioOpener,
    shutdown: asyncio.Event,
    tasks: list[asyncio.Task],
    connect_timeout: float,
    backoff: float,
    max_backoff: float,
    log: Callable[[str], Any],
) -> None:
    """Reopen ``bio``'s session each time it closes, until the run tears down.

    ``holder`` is the task holding the live session. It ends when the session fails, when a call
    finds it unusable, or at teardown; the session is then dropped and, unless the run is ending,
    reopened after ``backoff`` seconds, the wait doubling after each failed attempt up to
    ``max_backoff``. Detached like the holders, and like them it never raises: anything unexpected
    is logged and ends the reopening, which leaves the server's tools cut as they were before it.
    """
    try:
        while True:
            session = bio.session
            await asyncio.wait([holder])
            bio._drop(session)
            if shutdown.is_set():
                return
            log(f"[bio] the session to {bio.url} closed; reopening it")
            delay = backoff
            while True:
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=delay)
                    return  # the run is ending
                except asyncio.TimeoutError:
                    pass
                lost = asyncio.Event()
                session, task, reason = await _connect(
                    bio.url,
                    opener=opener,
                    shutdown=shutdown,
                    lost=lost,
                    tasks=tasks,
                    connect_timeout=connect_timeout,
                    retries=0,
                    backoff=0.0,
                    log=log,
                )
                if task is not None:
                    holder = task
                    break
                delay = min(2 * delay, max_backoff)
                log(f"[bio] could not reopen {bio.url} ({reason}); next attempt in {delay:g}s")
            bio._open(session, lost)
            log(f"[bio] reopened the session to {bio.url} (session {bio.opened})")
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise  # teardown cancel / interpreter exit — never swallowed
    except BaseException as exc:  # noqa: BLE001 — isolation boundary, as in the holders
        log(
            f"[bio] WARNING: reopening the session to {bio.url} failed ({_summarize_exc(exc)}); "
            "its tools stay cut for the rest of the run."
        )


@contextlib.asynccontextmanager
async def open_bio_sessions(
    urls: "tuple[str, ...] | list[str]",
    *,
    opener: BioOpener = _default_bio_opener,
    connect_timeout: float = _BIO_CONNECT_TIMEOUT_S,
    retries: int = _BIO_CONNECT_RETRIES,
    backoff: float = _BIO_CONNECT_BACKOFF_S,
    teardown_grace: float = _BIO_TEARDOWN_GRACE_S,
    reopen_max_backoff: float = _BIO_REOPEN_MAX_BACKOFF_S,
    log: Callable[[str], Any] = print,
) -> "AsyncIterator[list[Any]]":
    """Yield a list of LIVE, ISOLATED bio-knowledge MCP sessions (possibly empty).

    Each URL is opened in its own supervised :func:`_bio_session_runner` task, so a bio-session
    transport failure — at connect OR at any later point — degrades the biocurator instead of
    aborting the run (issue #86). Connect is best-effort with a per-attempt deadline and a couple
    of retries; an unreachable/slow server is warned and skipped. A hosted server's session is
    yielded as its :class:`BioSession`, which is reopened whenever the session closes
    (:func:`_keep_open`); the sessions are torn down on exit.

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
    # Every holder and keeper task ever created — all retrieved at teardown.
    tasks: list[asyncio.Task] = []
    sessions: list[Any] = []
    handles: list[BioSession] = []
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
                log(
                    f"[bio] OmniPath served LOCALLY from the {info.get('fetched_at')} snapshot "
                    f"({counts} rows) — no network, no rate limit"
                )
                continue
            lost = asyncio.Event()
            session, holder, reason = await _connect(
                url,
                opener=opener,
                shutdown=shutdown,
                lost=lost,
                tasks=tasks,
                connect_timeout=connect_timeout,
                retries=retries,
                backoff=backoff,
                log=log,
            )
            if holder is None:
                # Not kept open: its tools are discovered once, at startup, so a server first
                # reached later would have none to give back.
                log(
                    f"[bio] WARNING: could not connect to bio-knowledge server {url} "
                    f"({reason}); skipping it."
                )
                continue
            log(f"[bio] biocurator knowledge server connected: {url}")
            bio = BioSession(url, log=log)
            bio._open(session, lost)
            handles.append(bio)
            tasks.append(
                asyncio.create_task(
                    _keep_open(
                        bio,
                        holder,
                        opener=opener,
                        shutdown=shutdown,
                        tasks=tasks,
                        connect_timeout=connect_timeout,
                        backoff=backoff,
                        max_backoff=reopen_max_backoff,
                        log=log,
                    )
                )
            )
            # Read-through disk cache in front of the hosted server, wrapped HERE so every
            # consumer gets it and none has to know. A sweep asks the same SCENARIO-level
            # questions on every seed (each fujita run looks up the same four entities), so
            # after one warm run the rest are served from disk and never leave the cluster —
            # which is what stops the fair-use throttle being the limit on seed count, and
            # therefore on whether a sweep has the power to measure anything. On the handle, not
            # the session, so it outlives a reopen.
            cache = bio_cache.wrap_session(bio, url, log=log)
            if cache.enabled:
                caches.append(cache)
            sessions.append(bio)
        if caches:
            log(f"[bio] response cache: {bio_cache.cache_dir()}")
        try:
            yield sessions
        finally:
            for bio in handles:
                if bio.opened > 1:
                    log(f"[bio] the session to {bio.url} was reopened {bio.opened - 1} time(s)")
            # Report the HIT RATE, not just the path. Without this a sweep logs
            # "[bio] response cache: <path>" identically whether the cache is serving everything or
            # nothing -- and it was serving nothing: a blanket kwargs bypass meant a 0% hit rate in
            # production that no log line and no test could reveal. A number here is the thing that
            # would have shown it on the first run.
            for c in caches:
                with contextlib.suppress(Exception):
                    s = c.stats()
                    log(
                        f"[bio] cache {s['url']}: {s['hits']} hits / {s['misses']} misses "
                        f"(rate {s['hit_rate']}), {s['writes']} written, "
                        f"{s['bypasses']} bypassed, {s['errors']} errors"
                    )
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
                    asyncio.gather(*tasks, return_exceptions=True), timeout=teardown_grace
                )
