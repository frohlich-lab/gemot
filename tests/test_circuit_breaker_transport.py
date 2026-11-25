"""The circuit breaker must DEGRADE a transport error (a closed/hung MCP server — in practice a
flaky hosted bio server) immediately, not re-raise it: a propagated ClosedResourceError used to
abort the whole run (9 runs across sweeps died as `stop_reason: ClosedResourceError`).

Verified against the REAL anyio.ClosedResourceError so the by-name match in _is_transport_error is
pinned to the actual class. langchain_core is a harness dep, so skip where absent.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")
anyio = pytest.importorskip("anyio")
httpx2 = pytest.importorskip("httpx2")

from gemot.agent import circuit_breaker as _breaker  # noqa: E402


class _Tool:
    def __init__(self, name, coro):
        self.name, self.coroutine = name, coro


def _degraded_text(result):
    content, _artifact = result
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def test_is_transport_error_matches_real_and_generic():
    mod = _breaker
    assert mod._is_transport_error(anyio.ClosedResourceError()) is True  # the real class name
    assert mod._is_transport_error(ConnectionError("down")) is True
    # The WHOLE httpx2.TransportError family classifies as transport (same surface as the resume
    # backstop), including subclasses the old hand-listed name set omitted — e.g. PoolTimeout.
    # httpx2, which mcp 2 runs its transport on: an httpx error would derive from neither.
    assert mod._is_transport_error(httpx2.ConnectError("refused")) is True
    assert mod._is_transport_error(httpx2.PoolTimeout("no conn")) is True
    assert mod._is_transport_error(ValueError("bad args")) is False
    assert (
        mod._is_transport_error(
            httpx2.HTTPStatusError(  # NOT transport: a real HTTP response
                "500", request=httpx2.Request("GET", "http://x"), response=httpx2.Response(500)
            )
        )
        is False
    )


def test_breaker_degrades_a_transport_error_without_raising_or_3_strikes():
    """A single real ClosedResourceError on an EXTERNAL bio tool (named in transport_disable_names)
    disables it and returns the degraded message on the FIRST failure (no re-raise, no waiting for
    the 3-strike breaker)."""
    mod = _breaker
    calls = {"n": 0}

    async def _coro(**kw):
        calls["n"] += 1
        raise anyio.ClosedResourceError()

    state = mod._new_breaker_state()
    tool = _Tool("get_reactome_info_by_identifier", _coro)
    mod._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_reactome_info_by_identifier"}
    )

    result = asyncio.run(tool.coroutine())  # must NOT raise
    assert "TOOL DISABLED" in _degraded_text(result)
    # No session is reopening behind it, so it is gone for good, and the notice says so.
    assert "rest of this run" in _degraded_text(result)
    assert "get_reactome_info_by_identifier" in state["unreachable"]  # gone for every caller
    assert calls["n"] == 1  # called once, then degraded
    # A later call short-circuits on the disabled set (still no server hit, still no raise).
    assert "TOOL DISABLED" in _degraded_text(asyncio.run(tool.coroutine()))
    assert calls["n"] == 1


def test_transport_error_on_a_core_tool_is_not_auto_disabled():
    """A transport error on a CORE petab tool (NOT in transport_disable_names) must fall through to
    the normal path: re-raise on the first failure and leave the tool ENABLED — a single localhost
    blip must not silently strip `fit` / `set_model` for the rest of the run."""
    mod = _breaker

    async def _coro(**kw):
        raise anyio.ClosedResourceError()

    state = mod._new_breaker_state()
    tool = _Tool("fit", _coro)
    mod._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_reactome_info_by_identifier"}
    )
    with pytest.raises(anyio.ClosedResourceError):
        asyncio.run(tool.coroutine())
    assert "fit" not in state["disabled"] | state["unreachable"]


def test_breaker_still_reraises_a_normal_tool_error():
    """A NORMAL (non-transport) exception is unchanged: it re-raises (the model's retry-feedback
    path) and does not disable the tool on the first failure."""
    mod = _breaker

    async def _coro(**kw):
        raise ValueError("bad args")

    state = mod._new_breaker_state()
    tool = _Tool("some_tool", _coro)
    mod._wrap_tools_with_circuit_breaker([tool], state)
    with pytest.raises(ValueError):
        asyncio.run(tool.coroutine())
    assert "some_tool" not in state["disabled"] | state["unreachable"]


class _Session:
    """What a `BioSession` reaches: a live mcp session, here one that can be closed."""

    def __init__(self) -> None:
        self.closed = False

    async def call_tool(self, name, arguments=None):
        if self.closed:
            raise ConnectionError("Connection closed")
        return ([{"type": "text", "text": "an answer"}], None)


def test_a_bio_tool_cut_on_a_closed_session_comes_back_when_it_is_reopened():
    """frohlich-lab/PEtab-MCP#337 (C25): a closed hosted session cut four tools for 26 h, because
    nothing reopened it. Now the session is reopened (gemot.bio_sessions), the tool is given back
    for every caller once it is, and the notice in between says it is coming back rather than that
    it is gone for the rest of the run."""
    from gemot import bio_sessions

    handle = bio_sessions.BioSession("https://bio.example/mcp", log=lambda *_: None)
    handle._open(first := _Session(), asyncio.Event())
    calls = {"n": 0}

    async def _coro(**kw):
        calls["n"] += 1
        return await handle.call_tool("get_string_interactions", kw)

    state = _breaker._new_breaker_state()
    tool = _Tool("get_string_interactions", _coro)
    _breaker._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_string_interactions"}
    )
    first.closed = True

    cut = _degraded_text(asyncio.run(tool.coroutine()))
    assert cut.startswith("TOOL DISABLED") and "being reopened" in cut
    assert "rest of this run" not in cut
    assert _breaker.disabled_tools(state) == ["get_string_interactions"]
    # Cut for every caller until then: no server hit, the same notice.
    assert _degraded_text(asyncio.run(tool.coroutine())) == cut
    assert calls["n"] == 1

    handle._open(_Session(), asyncio.Event())  # the keeper's reopen

    assert _breaker.disabled_tools(state) == []
    assert asyncio.run(tool.coroutine())[0][0]["text"] == "an answer"
    assert calls["n"] == 2
    assert any("restored `get_string_interactions`" in e for e in state["events"])


def test_a_tool_cut_again_after_its_restore_comes_back_again():
    """A server can go away more than once in a run of days; each reopen gives the tool back."""
    from gemot import bio_sessions

    handle = bio_sessions.BioSession("https://bio.example/mcp", log=lambda *_: None)

    async def _coro(**kw):
        return await handle.call_tool("get_string_interactions", kw)

    state = _breaker._new_breaker_state()
    tool = _Tool("get_string_interactions", _coro)
    _breaker._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_string_interactions"}
    )
    for _ in range(2):
        handle._open(session := _Session(), asyncio.Event())
        session.closed = True
        asyncio.run(tool.coroutine())
        assert "get_string_interactions" in state["unreachable"]
    handle._open(_Session(), asyncio.Event())
    assert state["unreachable"] == set() and state["reopening"] == set()
