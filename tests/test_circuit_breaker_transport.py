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
httpx = pytest.importorskip("httpx")

from petab_harness.agent import circuit_breaker as _breaker  # noqa: E402


def _load_agent():
    return _breaker


class _Tool:
    def __init__(self, name, coro):
        self.name, self.coroutine = name, coro


def _degraded_text(result):
    content, _artifact = result
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def test_is_transport_error_matches_real_and_generic():
    mod = _load_agent()
    assert mod._is_transport_error(anyio.ClosedResourceError()) is True   # the real class name
    assert mod._is_transport_error(ConnectionError("down")) is True
    # The WHOLE httpx.TransportError family classifies as transport (same surface as the resume
    # backstop), including subclasses the old hand-listed name set omitted — e.g. PoolTimeout.
    assert mod._is_transport_error(httpx.ConnectError("refused")) is True
    assert mod._is_transport_error(httpx.PoolTimeout("no conn")) is True
    assert mod._is_transport_error(ValueError("bad args")) is False
    assert mod._is_transport_error(httpx.HTTPStatusError(  # NOT transport: a real HTTP response
        "500", request=httpx.Request("GET", "http://x"),
        response=httpx.Response(500))) is False


def test_breaker_degrades_a_transport_error_without_raising_or_3_strikes():
    """A single real ClosedResourceError on an EXTERNAL bio tool (named in transport_disable_names)
    disables it and returns the degraded message on the FIRST failure (no re-raise, no waiting for
    the 3-strike breaker)."""
    mod = _load_agent()
    calls = {"n": 0}

    async def _coro(runtime=None, **kw):
        calls["n"] += 1
        raise anyio.ClosedResourceError()

    state = mod._new_breaker_state()
    tool = _Tool("get_reactome_info_by_identifier", _coro)
    mod._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_reactome_info_by_identifier"})

    result = asyncio.run(tool.coroutine(runtime=None))          # must NOT raise
    assert "TOOL DISABLED" in _degraded_text(result)
    assert "get_reactome_info_by_identifier" in state["disabled"]
    assert calls["n"] == 1                                       # called once, then degraded
    # A later call short-circuits on the disabled set (still no server hit, still no raise).
    assert "TOOL DISABLED" in _degraded_text(asyncio.run(tool.coroutine(runtime=None)))
    assert calls["n"] == 1


def test_transport_error_on_a_core_tool_is_not_auto_disabled():
    """A transport error on a CORE petab tool (NOT in transport_disable_names) must fall through to
    the normal path: re-raise on the first failure and leave the tool ENABLED — a single localhost
    blip must not silently strip `fit` / `set_model` for the rest of the run."""
    mod = _load_agent()

    async def _coro(runtime=None, **kw):
        raise anyio.ClosedResourceError()

    state = mod._new_breaker_state()
    tool = _Tool("fit", _coro)
    mod._wrap_tools_with_circuit_breaker([tool], state,
                                         transport_disable_names={"get_reactome_info_by_identifier"})
    with pytest.raises(anyio.ClosedResourceError):
        asyncio.run(tool.coroutine(runtime=None))
    assert "fit" not in state["disabled"]


def test_breaker_still_reraises_a_normal_tool_error():
    """A NORMAL (non-transport) exception is unchanged: it re-raises (the model's retry-feedback
    path) and does not disable the tool on the first failure."""
    mod = _load_agent()

    async def _coro(runtime=None, **kw):
        raise ValueError("bad args")

    state = mod._new_breaker_state()
    tool = _Tool("some_tool", _coro)
    mod._wrap_tools_with_circuit_breaker([tool], state)
    with pytest.raises(ValueError):
        asyncio.run(tool.coroutine(runtime=None))
    assert "some_tool" not in state["disabled"]
