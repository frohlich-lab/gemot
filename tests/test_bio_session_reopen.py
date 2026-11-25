"""A hosted bio session that closes mid-run is reopened (gemot.bio_sessions).

Under mcp 2 a hosted server gone away ends the session for good: every later call raises
``MCPError("Connection closed")``, even once the server is back, and nothing opened a new one. The
circuit breaker cut the tools on the first such call and they stayed cut: four tools for 26 h in one
run of campaign gemot_pools_20261107 (frohlich-lab/PEtab-MCP#337, finding C25). The run now holds a
``BioSession`` per server, which a keeper task reopens each time its session closes.

The first part drives the supervision with a fake server, as tests/test_bio_session_isolation.py
does. The last test runs the whole path against a real fastmcp server over streamable HTTP that is
killed and restarted, so a change in how mcp reports a closed session fails here.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time

import pytest

from gemot import bio_sessions as bio

_QUIET = {"log": lambda *_: None}


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """The fake sessions answer `call_tool`, so the cache would wrap them and write its entries
    into the repository's own cache directory."""
    monkeypatch.setenv("PETAB_BIO_CACHE_DIR", "")


class _FakeSession:
    """One session to the fake server; ``killed`` ends it the way mcp's transport does."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.closed = False
        self.killed = asyncio.Event()

    async def crash_when_killed(self) -> None:
        await self.killed.wait()
        self.closed = True
        raise ConnectionError("All connection attempts failed")

    async def call_tool(self, name, arguments=None):
        if self.closed:
            raise ConnectionError("Connection closed")
        if name == "bad":
            raise ValueError("bad arguments")
        return f"{name} from session {self.n}"

    async def list_tools(self, *args, **kwargs):
        return ["ping"]


class _Server:
    """A fake hosted server. ``opener`` connects unless it is ``down``; a session's transport task
    runs beside the holder in a task group, so killing the session cancels the holder and raises an
    exception group out of it, as anyio's group does for mcp's transport."""

    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []
        self.down = False
        self.connects = 0

    def opener(self, url):
        @contextlib.asynccontextmanager
        async def _open():
            self.connects += 1
            if self.down:
                raise ConnectionError("refused")
            session = _FakeSession(len(self.sessions) + 1)
            self.sessions.append(session)
            async with asyncio.TaskGroup() as tg:
                transport = tg.create_task(session.crash_when_killed())
                try:
                    yield session
                finally:
                    transport.cancel()

        return _open()


async def _eventually(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


def _open(server: _Server, **kwargs):
    kwargs = {"backoff": 0.01, "reopen_max_backoff": 0.05, "teardown_grace": 0.5, **kwargs}
    return bio.open_bio_sessions(["u"], opener=server.opener, **_QUIET, **kwargs)


def test_a_session_whose_transport_fails_is_reopened_behind_the_same_handle():
    """The mcp 2 shape: the server goes away, the transport's error ends the holder, and the session
    is dead. The handle the tools were built on gets a new one."""
    server = _Server()

    async def go():
        async with _open(server) as (handle,):
            first = await handle.call_tool("ping", {})
            server.sessions[0].killed.set()
            await _eventually(lambda: handle.opened == 2)
            return first, await handle.call_tool("ping", {})

    assert asyncio.run(go()) == ("ping from session 1", "ping from session 2")


def test_a_call_that_finds_the_session_closed_drops_it_and_reopens():
    """A session can also close with its holder still parked on it (the transport's writer died, and
    the receive loop saw end of stream). Only a call finds out; it drops the session, which wakes
    the holder to close it and the keeper to reopen it."""
    server = _Server()

    async def go():
        async with _open(server) as (handle,):
            server.sessions[0].closed = True
            with pytest.raises(bio.BioSessionClosed) as raised:
                await handle.call_tool("ping", {})
            await _eventually(lambda: handle.opened == 2)
            return raised.value, await handle.call_tool("ping", {}), handle

    error, after, handle = asyncio.run(go())
    assert error.session is handle
    assert isinstance(error, ConnectionError)  # the breaker's transport test matches it
    assert "Connection closed" in str(error)
    assert after == "ping from session 2"


def test_a_call_while_the_session_is_being_reopened_raises_closed():
    server = _Server()

    async def go():
        async with _open(server, backoff=10.0) as (handle,):
            server.sessions[0].killed.set()
            await _eventually(lambda: handle.session is None)
            with pytest.raises(bio.BioSessionClosed):
                await handle.call_tool("ping", {})
            with pytest.raises(bio.BioSessionClosed):
                await handle.list_tools()

    asyncio.run(go())


def test_a_tool_error_passes_through_and_keeps_the_session():
    """Only a transport error says the session is no use; an argument error is the model's."""
    server = _Server()

    async def go():
        async with _open(server) as (handle,):
            with pytest.raises(ValueError, match="bad arguments"):
                await handle.call_tool("bad", {})
            return handle.session is server.sessions[0], handle.opened

    assert asyncio.run(go()) == (True, 1)


def test_a_server_that_stays_down_is_retried_until_it_is_back():
    server = _Server()

    async def go():
        async with _open(server) as (handle,):
            server.down = True
            server.sessions[0].killed.set()
            await _eventually(lambda: server.connects >= 4)
            assert handle.session is None
            server.down = False
            await _eventually(lambda: handle.opened == 2)
            return await handle.call_tool("ping", {})

    assert asyncio.run(go()) == "ping from session 2"


def test_the_backoff_doubles_up_to_its_cap():
    logs: list[str] = []
    server = _Server()

    async def go():
        async with bio.open_bio_sessions(
            ["u"], opener=server.opener, backoff=0.01, reopen_max_backoff=0.04, log=logs.append
        ) as (handle,):
            server.down = True
            server.sessions[0].killed.set()
            await _eventually(lambda: server.connects >= 5)

    asyncio.run(go())
    waits = [m.rsplit(" ", 1)[1] for m in logs if "next attempt in" in m]
    assert waits[:3] == ["0.02s", "0.04s", "0.04s"]


def test_teardown_is_prompt_while_a_reopen_is_waiting():
    """A keeper backing off must not hold the run's teardown for its wait."""
    server = _Server()

    async def go():
        async with _open(server, backoff=60.0, teardown_grace=5.0) as (handle,):
            server.sessions[0].killed.set()
            await _eventually(lambda: handle.session is None)
            t0 = time.monotonic()
        return time.monotonic() - t0

    assert asyncio.run(go()) < 1.0


def test_when_open_calls_back_now_or_at_the_reopen():
    server = _Server()
    calls: list[str] = []

    async def go():
        async with _open(server) as (handle,):
            handle.when_open(lambda: calls.append("open now"))
            server.sessions[0].killed.set()
            await _eventually(lambda: handle.session is None)
            handle.when_open(lambda: calls.append("at the reopen"))
            assert calls == ["open now"]
            await _eventually(lambda: handle.opened == 2)

    asyncio.run(go())
    assert calls == ["open now", "at the reopen"]


def test_the_cache_shadow_outlives_the_session(tmp_path, monkeypatch):
    """The cache shadows the handle's `call_tool`, so a reopen keeps it, and its hit rate counts
    the whole run. An answer it holds is served while the server is down."""
    monkeypatch.setenv("PETAB_BIO_CACHE_DIR", str(tmp_path))
    server = _Server()

    async def go():
        async with _open(server, backoff=10.0) as (handle,):
            first = await handle.call_tool("ping", {})
            server.sessions[0].killed.set()
            await _eventually(lambda: handle.session is None)
            return first, await handle.call_tool("ping", {})

    first, while_down = asyncio.run(go())
    assert first == while_down == "ping from session 1"


# --- against a real server ----------------------------------------------------------------------

_SERVER = r"""
import sys
from fastmcp import FastMCP
srv = FastMCP("bio")

@srv.tool()
def ping() -> str:
    return "pong"

srv.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
"""


def test_a_real_bio_server_killed_and_restarted_gets_its_tool_back(tmp_path, monkeypatch):
    """The whole path on mcp's own transport: the tool bridge over the handle, the breaker cutting
    the tool when the server dies, the keeper reopening once it is back, and the tool restored."""
    pytest.importorskip("fastmcp", reason="dev dependency")
    pytest.importorskip("langchain.mcp", reason="agent stack not installed")
    from gemot.agent import circuit_breaker as B
    from gemot.agent.mcp_tools import load_mcp_tools
    from gemot.runner.server import _free_port, _wait_until_listening

    monkeypatch.setenv("PETAB_BIO_CACHE_DIR", "")
    port, log = _free_port(), tmp_path / "server.log"

    def start():
        proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER, str(port)],
            stdout=subprocess.DEVNULL,
            stderr=log.open("w"),
        )
        _wait_until_listening("127.0.0.1", port, proc, log, timeout=60)
        return proc

    def text(result):
        content, _ = result
        return " ".join(b["text"] for b in content if isinstance(b, dict))

    procs = [start()]

    async def go():
        url = f"http://127.0.0.1:{port}/mcp"
        async with bio.open_bio_sessions(
            [url], backoff=0.1, reopen_max_backoff=0.3, teardown_grace=1.0, **_QUIET
        ) as (handle,):
            (tool,) = await load_mcp_tools(handle)
            state = B._new_breaker_state()
            B._wrap_tools_with_circuit_breaker([tool], state, transport_disable_names={"ping"})
            before = text(await tool.coroutine())

            procs[0].kill()
            procs[0].wait(timeout=10)
            cut = text(await tool.coroutine())
            lost = sorted(B.disabled_tools(state))

            procs.append(start())
            await _eventually(lambda: "ping" not in state["unreachable"], timeout=30)
            return before, cut, lost, text(await tool.coroutine()), handle.opened

    try:
        before, cut, lost, after, opened = asyncio.run(go())
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=10)

    assert before == "pong"
    assert cut.startswith("TOOL DISABLED") and "being reopened" in cut
    assert lost == ["ping"]
    assert after == "pong"
    assert opened == 2
