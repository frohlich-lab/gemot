"""A tool call lives on the server's heartbeats, not on a budget sized from its fit (gemot#106).

On a shared fit pool a fit's wall time includes its wait behind other runs' fits, which no formula
bounds, so the waves-times-cap budget `_wrap_call_timeout` gave each call would cut a fit that is
only queued. The server sends a progress notification on every call every 30 s (PEtab-MCP#224),
and a call is cut once it has gone `CALL_IDLE_WINDOW_S` without a live one.

The plan's Check 9 runs here against an in-memory server whose tool sends heartbeats in
PEtab-MCP#224's format, over a real ClientSession in the handshake era the runner speaks, with the
window shrunk to fractions of a second. That the server is not told to stop the fit it was cut on
is checked over streamable HTTP, with the runner's own dispatcher, in
tests/test_abandoned_fit_runs_on.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time

import pytest

from gemot import infrastructure as I
from gemot import prompts
from gemot.agent import circuit_breaker as B
from gemot.agent import driver as D
from gemot.fit_budget import FIT_COMPILE_MARGIN_S
from gemot.runner import clock as C

pytest.importorskip("mcp", reason="harness-only dependency")

from mcp import MCPError  # noqa: E402
from mcp.server.mcpserver import Context  # noqa: E402 - resolved from here by the test server
from mcp.types import REQUEST_TIMEOUT  # noqa: E402

#: The window the tests run at, and the interval the test server beats at inside it.
_WINDOW_S = 0.5
_BEAT_S = 0.1

_QUEUED = "queued for 3600s, 0 of 4 starts done"
_STUCK = f"compiling for {FIT_COMPILE_MARGIN_S + 1:.0f}s"


@pytest.fixture(autouse=True)
def _short_window(monkeypatch):
    monkeypatch.setattr(C, "CALL_IDLE_WINDOW_S", _WINDOW_S)


def _server():
    """A one-tool server whose `make_changes` sends one heartbeat per entry of `beats`, `_BEAT_S`
    apart, as PEtab-MCP's `_Heartbeat` does, then stays silent for `silent` seconds and returns.
    ``{age}`` in a heartbeat is the call's seconds so far, as the server would count its phase."""
    import anyio
    from mcp.server import MCPServer

    srv = MCPServer("heartbeats")

    @srv.tool()
    async def make_changes(ctx: Context, beats: list[str], silent: float = 0.0) -> str:
        began = time.monotonic()
        for message in beats:
            await anyio.sleep(_BEAT_S)
            age = time.monotonic() - began
            await ctx.report_progress(age, message=message.replace("{age}", f"{age:.0f}"))
        await anyio.sleep(silent)
        return "fitted"

    return srv


def _call(arguments, **kwargs):
    """``make_changes(arguments)`` on `_server()` through the heartbeat timeout: the result or the
    error it raised, the seconds it took, the run's heartbeats, and every `read_timeout_seconds`
    that reached mcp beneath the shadow."""
    from mcp import Client

    async def go():
        async with Client(_server(), mode="legacy") as client:
            budgets, original = [], client.session.call_tool

            async def spy(*args, **kw):
                budgets.append(kw.get("read_timeout_seconds"))
                return await original(*args, **kw)

            client.session.call_tool = spy
            heartbeats = C._wrap_call_timeout(client.session)
            began = time.monotonic()
            try:
                out = await client.session.call_tool("make_changes", arguments, **kwargs)
            except Exception as exc:  # noqa: BLE001 - the test inspects it
                out = exc
            return out, time.monotonic() - began, heartbeats, budgets

    return asyncio.run(go())


@contextlib.asynccontextmanager
async def _session():
    """A real ClientSession on `_server()`, with the heartbeat timeout installed."""
    from mcp import Client

    async with Client(_server(), mode="legacy") as client:
        C._wrap_call_timeout(client.session)
        yield client.session


# -- Check 9 ---------------------------------------------------------------------------------
def test_a_fit_queued_for_many_windows_lives_on_its_heartbeats_and_returns():
    """The plan's first check: a fit queued for longer than any budget the call was given keeps
    its call alive and returns its result. The call reaches mcp with no budget at all, so nothing
    but the heartbeats can be keeping it."""
    beats = ["queued for {age}s, 0 of 4 starts done"] * round(4 * _WINDOW_S / _BEAT_S)
    heard = []

    async def mine(progress, total, message):
        heard.append(message)

    result, took, heartbeats, budgets = _call({"beats": beats}, progress_callback=mine)
    assert C._unwrap(result) == "fitted"
    assert took > 3 * _WINDOW_S, "the call did not outlive the window, so this proves nothing"
    assert budgets == [math.inf]
    # The caller's own progress callback still hears every heartbeat (the adapter's, in the panel).
    assert len(heard) == len(beats) and all(m.startswith("queued for") for m in heard)
    # ...and the wait counts as queue time, which the driver leaves out of the wall budget.
    assert 0 < heartbeats.queue_seconds <= took


def test_a_silent_server_ends_the_call_within_the_window():
    """The plan's second check, first half: no heartbeat in the window ends the call with the
    error mcp's own clock raised, so every classifier downstream reads it as it did."""
    error, took, _, _ = _call({"beats": [], "silent": 10.0})
    assert isinstance(error, MCPError) and error.code == REQUEST_TIMEOUT
    assert B._is_call_timeout(error)
    assert "no heartbeat" in str(error)
    assert _WINDOW_S <= took < 4 * _WINDOW_S


def test_heartbeats_that_stop_end_the_call_one_window_after_the_last():
    beats = [_QUEUED] * round(3 * _WINDOW_S / _BEAT_S)
    error, took, _, _ = _call({"beats": beats, "silent": 10.0})
    assert isinstance(error, MCPError) and error.code == REQUEST_TIMEOUT
    assert 3.9 * _WINDOW_S <= took < 7 * _WINDOW_S


def test_a_silent_server_takes_the_wedged_fit_path():
    """The plan's second check, second half: the cut reaches the lead as the fit-timeout note, and
    a second cut behind it stops the run as `infrastructure_error: fit_wedged`, as a fit that
    outran its budget did before (gemot#56)."""

    async def go():
        async with _session() as session:

            class _Fit:
                name = "make_changes"

                @staticmethod
                async def coroutine(**arguments):
                    return await session.call_tool("make_changes", arguments)

            fit = _Fit()
            state = B._new_breaker_state()
            B._wrap_tools_with_circuit_breaker([fit], state, fit_tools=frozenset({"make_changes"}))
            note, _ = await fit.coroutine(beats=[], silent=10.0)
            with pytest.raises(I.InfrastructureError) as raised:
                await fit.coroutine(beats=[], silent=10.0)
            return note, raised.value.evidence

    note, evidence = asyncio.run(go())
    assert note.startswith(prompts.FIT_CALL_TIMEOUT_TMPL.split("{name}")[0])
    assert "no heartbeat" in note
    assert evidence.cause == I.FIT_WEDGED


def test_a_compile_past_its_margin_ends_the_call():
    """The plan's third check: the server keeps beating while a build hangs, so a heartbeat that
    shows a compile past `FIT_COMPILE_MARGIN_S` counts as none."""
    error, took, _, _ = _call({"beats": [_STUCK] * round(6 * _WINDOW_S / _BEAT_S)})
    assert isinstance(error, MCPError) and error.code == REQUEST_TIMEOUT
    assert _STUCK in str(error)
    assert took < 4 * _WINDOW_S


def test_a_compile_inside_its_margin_is_alive():
    """The control for the check above: compiling is a phase like any other until the margin."""
    result, took, _, _ = _call({"beats": ["compiling for 10s"] * round(3 * _WINDOW_S / _BEAT_S)})
    assert C._unwrap(result) == "fitted"
    assert took > 2 * _WINDOW_S


# -- the call's arguments --------------------------------------------------------------------
class _FakeSession:
    """Records the arguments each call_tool reached the transport with, and replays `beats` to
    whatever progress callback it was handed."""

    def __init__(self, *beats):
        self.calls, self.beats = [], beats

    async def call_tool(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        callback = args[3] if len(args) > 3 else kwargs.get("progress_callback")
        for i, message in enumerate(self.beats):
            await callback(float(i), None, message)
        return {}


def test_an_explicit_caller_timeout_is_kept():
    s = _FakeSession()
    C._wrap_call_timeout(s)
    asyncio.run(s.call_tool("make_changes", {}, read_timeout_seconds=7))
    assert s.calls[0][1]["read_timeout_seconds"] == 7


def test_positional_arguments_are_not_given_twice():
    """call_tool's third and fourth positionals ARE read_timeout_seconds and progress_callback;
    a keyword on top of either raises "got multiple values for argument" — the wrapper failing
    the call it is protecting."""
    heard = []

    async def mine(progress, total, message):
        heard.append(message)

    s = _FakeSession(_QUEUED)
    C._wrap_call_timeout(s)
    asyncio.run(s.call_tool("make_changes", {}, 9, mine))
    args, kwargs = s.calls[0]
    assert args[2] == 9 and "read_timeout_seconds" not in kwargs
    assert "progress_callback" not in kwargs
    assert heard == [_QUEUED]


def test_other_kwargs_are_passed_through_untouched():
    sentinel = object()
    s = _FakeSession()
    C._wrap_call_timeout(s)
    asyncio.run(s.call_tool("make_changes", {"n_starts": 400}, meta=sentinel))
    args, kwargs = s.calls[0]
    assert args == ("make_changes", {"n_starts": 400})
    assert kwargs["meta"] is sentinel


# -- queue time, counted once ----------------------------------------------------------------
def test_queue_time_is_the_union_of_every_calls_heartbeats(monkeypatch):
    """A read waiting behind a fit gets the fit's heartbeats, so summing per call would count the
    wait twice; a wait that began after the last heartbeat must not be backdated to it."""
    now = [0.0]
    monkeypatch.setattr(C.time, "monotonic", lambda: now[0])
    heartbeats = C._Heartbeats()

    def at(t, message):
        now[0] = t
        return heartbeats.hear(message)

    at(0.0, "queued for 30s, 0 of 4 starts done")  # the wait so far: 30
    at(30.0, "queued for 60s, 0 of 4 starts done")  # +30
    at(31.0, "queued for 61s")  # a read beside the fit: +1, not +61
    at(60.0, "running for 5s, 1 of 4 starts done")  # +0
    at(90.0, "queued for 10s, 1 of 4 starts done")  # a new wait: +10, not +30
    at(95.0, None)  # no change running
    assert heartbeats.queue_seconds == 71.0


def test_only_a_compile_past_its_margin_is_not_live():
    heartbeats = C._Heartbeats()
    assert heartbeats.hear(_QUEUED)
    assert heartbeats.hear("running for 900s, 3 of 4 starts done")
    assert heartbeats.hear(None)
    assert heartbeats.hear(f"compiling for {FIT_COMPILE_MARGIN_S:.0f}s")
    assert not heartbeats.hear(_STUCK)


# -- the stall watchdog, fed by the same heartbeats ------------------------------------------
async def _step_after(seconds):
    await asyncio.sleep(seconds)
    yield "stepped"


async def _next_step_while_beating(message, *, timeout, step_s, left=None):
    """`_next_step` on a step that takes `step_s`, while the server beats `message`."""
    heartbeats = C._Heartbeats()

    async def beat():
        while True:
            await asyncio.sleep(_BEAT_S)
            heartbeats.hear(message)

    beating = asyncio.create_task(beat())
    try:
        return await D._next_step(
            _step_after(step_s).__aiter__(), timeout, heartbeats=heartbeats, left=left
        )
    finally:
        beating.cancel()


def test_a_heartbeat_pushes_the_stall_watchdog_out():
    """A step whose fit is queued for the pool for hours is not a stall."""
    out = asyncio.run(_next_step_while_beating(_QUEUED, timeout=_WINDOW_S, step_s=4 * _WINDOW_S))
    assert out == "stepped"


def test_without_a_live_heartbeat_the_watchdog_still_fires():
    with pytest.raises(D._StepStalled):
        asyncio.run(_next_step_while_beating(_STUCK, timeout=_WINDOW_S, step_s=4 * _WINDOW_S))


def test_heartbeats_never_push_the_watchdog_past_the_wall_budget():
    """The wall budget stays the hard ceiling: a step still running at its end is handed back."""
    began = time.monotonic()
    with pytest.raises(D._StepStalled):
        asyncio.run(
            _next_step_while_beating(
                _QUEUED,
                timeout=10 * _WINDOW_S,
                step_s=10 * _WINDOW_S,
                left=lambda: 1.0 + began - time.monotonic(),
            )
        )
    assert time.monotonic() - began < 4 * _WINDOW_S + 1.0
