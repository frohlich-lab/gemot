"""Tests for the biocurator bio-MCP session isolation (petab_harness.bio_sessions).

Covers issue #86: a transport failure on a hosted bio-knowledge MCP session used to abort the
whole deepagents run. ``open_bio_sessions`` now runs each session in its own supervised, detached
``asyncio`` task, so a failure at connect OR mid-run degrades the biocurator instead of aborting
the run.

``bio_sessions`` imports ``mcp`` LAZILY (only inside the real opener), and ``petab_harness``'s own
``__init__`` is lazy too, so these tests just import it and inject a FAKE opener — no ``mcp``, no
``langchain``, no network, no live bio servers.
The supervision/teardown logic is pure ``asyncio`` and always runs; the end-to-end transport abort
can only be exercised against a live server (see tests/test_biocurator_tools.py, which skips
offline). Each async case is driven with ``asyncio.run`` so no pytest-asyncio config is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from petab_harness import bio_sessions as bio


# --------------------------------------------------------------------- fake sessions + openers
class _FakeSession:
    def __init__(self, name: str) -> None:
        self.name = name


def _opener_ok(name: str | None = None):
    """An opener that connects fast and stays open until torn down."""

    @contextlib.asynccontextmanager
    async def _o(url):
        await asyncio.sleep(0)
        yield _FakeSession(name or url)

    return _o


def _opener_connect_error(url):
    """An opener that fails DURING connect (before yielding a session)."""

    @contextlib.asynccontextmanager
    async def _o(u):
        raise RuntimeError("simulated ConnectError")
        yield  # pragma: no cover - unreachable, marks this a generator CM

    return _o


def _opener_hang(url):
    """An opener whose connect never completes (a wedged TLS handshake)."""

    @contextlib.asynccontextmanager
    async def _o(u):
        await asyncio.sleep(1000)
        yield _FakeSession(u)  # pragma: no cover

    return _o


def _opener_crash_at_close(url):
    """An opener that hands out a live session, then RAISES when the session closes — the
    mechanism by which a teardown-time bio failure surfaces (the anyio group re-raises at scope
    exit). The runner must swallow it."""

    @contextlib.asynccontextmanager
    async def _o(u):
        try:
            yield _FakeSession(u)
        finally:
            raise RuntimeError("simulated ClosedResourceError at session close")

    return _o


def _opener_capturing(captured: list):
    """An opener that records ITS OWN holder (runner) task into ``captured``, so a test can kill
    that task MID-RUN — simulating a live bio session dying while the run body still executes (the
    actual #86 scenario, which the background post_writer crash produces in production)."""

    @contextlib.asynccontextmanager
    async def _o(u):
        captured.append(asyncio.current_task())  # this coroutine runs inside the holder task
        yield _FakeSession(u)

    return _o


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- open_bio_sessions
def test_all_servers_connect_yields_all_sessions():
    async def go():
        async with bio.open_bio_sessions(["a", "b"], opener=_opener_ok(), log=lambda *_: None) as s:
            return [x.name for x in s]

    assert _run(go()) == ["a", "b"]


def test_empty_urls_yields_no_sessions():
    async def go():
        async with bio.open_bio_sessions([], opener=_opener_ok(), log=lambda *_: None) as s:
            return s

    assert _run(go()) == []


def test_one_connect_failure_is_skipped_others_kept_no_raise():
    def mixed(u):
        return _opener_connect_error(u)(u) if u == "bad" else _opener_ok()(u)

    async def go():
        # A connect failure must NOT propagate out of the context manager.
        async with bio.open_bio_sessions(
            ["bad", "good"], opener=mixed, retries=0, log=lambda *_: None
        ) as s:
            return [x.name for x in s]

    assert _run(go()) == ["good"]


def test_connect_hang_is_skipped_and_time_bounded():
    async def go():
        t0 = time.monotonic()
        # Explicit small teardown_grace so a hung holder can't let teardown dominate the wall time
        # (the default is 5s) — keeps the bound robust on a loaded CI runner.
        async with bio.open_bio_sessions(
            ["h"],
            opener=_opener_hang("h"),
            connect_timeout=0.2,
            retries=1,
            backoff=0.05,
            teardown_grace=0.2,
            log=lambda *_: None,
        ) as s:
            sessions = list(s)
        return sessions, time.monotonic() - t0

    sessions, elapsed = _run(go())
    assert sessions == []
    # 2 attempts x 0.2s deadline + 1 x 0.05s backoff + bounded teardown (<= 2 x 0.2s): the work is
    # ~0.65s, so a 3s bound tolerates substantial scheduler jitter without ever hanging.
    assert elapsed < 3.0, elapsed


def test_connect_retries_then_succeeds():
    state = {"n": 0}

    def flaky(u):
        state["n"] += 1
        return _opener_connect_error(u)(u) if state["n"] == 1 else _opener_ok("retried")(u)

    async def go():
        async with bio.open_bio_sessions(
            ["r"], opener=flaky, retries=2, backoff=0.01, log=lambda *_: None
        ) as s:
            return [x.name for x in s]

    assert _run(go()) == ["retried"]
    assert state["n"] == 2  # failed once, succeeded on the retry


def test_session_close_failure_never_propagates_out_of_context_manager():
    # The one residual path that could regress a SUCCESSFUL run to degraded: a bio session that
    # dies while being torn down. open_bio_sessions.__aexit__ must swallow it.
    async def go():
        async with bio.open_bio_sessions(
            ["c"], opener=_opener_crash_at_close("c"), log=lambda *_: None
        ) as s:
            assert len(s) == 1
        return "exited-cleanly"

    assert _run(go()) == "exited-cleanly"


def test_run_body_survives_a_bio_session_dying_mid_run():
    # THE core #86 guarantee: a bio holder task dying WHILE the run body executes must not cancel
    # the body. The holder is a detached sibling task, so killing it mid-run leaves the body — a
    # coroutine on the run's OWN task — untouched, and the run continues to completion.
    captured: list = []

    async def go():
        progress = []
        async with bio.open_bio_sessions(
            ["c"], opener=_opener_capturing(captured), teardown_grace=0.2, log=lambda *_: None
        ) as s:
            assert len(s) == 1 and captured, "session not handed out / holder not captured"
            captured[0].cancel()  # kill the live bio holder MID-RUN
            for i in range(3):
                await asyncio.sleep(0)  # the run body keeps working after the holder dies
                progress.append(i)
            assert captured[0].done()  # the holder died; the body did not
        return progress

    assert _run(go()) == [0, 1, 2]


def test_run_body_completes_despite_a_bio_session_failing_at_teardown():
    # Complements the mid-run test: a bio session that fails at TEARDOWN (its close raises) must
    # also leave the body's result intact and not propagate out of the context manager.
    async def go():
        progress = []
        async with bio.open_bio_sessions(
            ["c"], opener=_opener_crash_at_close("c"), log=lambda *_: None
        ):
            for i in range(3):
                await asyncio.sleep(0)  # simulate the run doing real work
                progress.append(i)
        return progress

    assert _run(go()) == [0, 1, 2]


# --------------------------------------------------------------------- _bio_session_runner
def test_runner_connect_failure_is_contained():
    async def go():
        holder = {"session": None, "error": None}
        ready, shutdown = asyncio.Event(), asyncio.Event()
        # Must return normally (swallowed), with the error reported via holder for the supervisor.
        await bio._bio_session_runner(
            "x", holder, ready, shutdown, _opener_connect_error("x"), lambda *_: None
        )
        return ready.is_set(), holder["session"], holder["error"]

    ready_set, session, error = _run(go())
    assert ready_set is True
    assert session is None
    assert isinstance(error, RuntimeError)


def test_runner_live_failure_is_swallowed():
    # A live-session failure AFTER hand-out (the #86 crash, surfaced as a raise at scope exit)
    # must be swallowed — the runner returns normally, never raising into the run's task.
    logs: list[str] = []

    async def go():
        holder = {"session": None, "error": None}
        ready, shutdown = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            bio._bio_session_runner(
                "c", holder, ready, shutdown, _opener_crash_at_close("c"), logs.append
            )
        )
        await asyncio.wait_for(ready.wait(), timeout=1)
        assert holder["session"] is not None  # session was handed out
        shutdown.set()  # let it exit -> the close raises inside it
        await asyncio.wait_for(task, timeout=1)  # must NOT raise
        return task.exception() is None

    assert _run(go()) is True
    assert any("failed after connect" in m for m in logs)


def test_runner_normal_path_returns_on_shutdown():
    async def go():
        holder = {"session": None, "error": None}
        ready, shutdown = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(
            bio._bio_session_runner("ok", holder, ready, shutdown, _opener_ok(), lambda *_: None)
        )
        await asyncio.wait_for(ready.wait(), timeout=1)
        assert not task.done()  # parked, holding the session open
        shutdown.set()
        await asyncio.wait_for(task, timeout=1)
        return holder["session"].name, task.exception() is None

    name, clean = _run(go())
    assert name == "ok"
    assert clean is True


# --------------------------------------------------------------------- helper
def test_summarize_exc_unwraps_exception_group():
    inner = ValueError("boom")
    group = BaseExceptionGroup("grp", [inner])
    assert bio._summarize_exc(group) == "ValueError: boom"
    assert bio._summarize_exc(RuntimeError("x")).startswith("RuntimeError: x")
