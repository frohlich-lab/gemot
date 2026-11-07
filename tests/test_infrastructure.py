"""A failure of the server's infrastructure ends the run, unjudged, with a label to resubmit.

gemot#39 / PEtab-MCP#83: in the INDRA-arm sweep the lab quota filled for a few hours and the server
reported it to the agent as ill-posed models. Runs kept going, leads re-authored models that were
fine, and 30 runs that completed carried the damage with no mark in result.json. These pin the
evidence (the server's own report since PEtab-MCP#95, and the raw disk-full text), the log watch,
the tool wrapper's stop, the judge skip, and how the analysis files it.

The driver loop's half is in tests/test_run_loop_no_model.py, next to the other loop exits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from gemot import analysis as A
from gemot import experiments
from gemot import infrastructure as I
from gemot.agent import circuit_breaker as B

# The shapes the sweep recorded (PEtab-MCP#83), with the paths cut.
_FIT_CRASH = "the edit applied but the fit crashed unexpectedly: [Errno 122] Disk quota exceeded"
_TOOL_ERROR = "Error calling tool 'make_changes': [Errno 122] Disk quota exceeded"
_LINK = "/usr/bin/ld: final link failed: Disk quota exceeded"
# The server's report since PEtab-MCP#95: a tool error whose text starts with the prefix.
_REPORT = "infrastructure_error: disk_full: [Errno 122] Disk quota exceeded: 'x.h5'"


@pytest.mark.parametrize(
    "text",
    [
        _FIT_CRASH,
        _TOOL_ERROR,
        _LINK,
        "OSError: [Errno 28] No space left on device: 'result.h5'",
        "HDF5-DIAG: ... errno = 122, error message = 'Disk quota exceeded'",
        "write failed: ENOSPC",
    ],
)
def test_a_full_disk_is_found_wherever_it_is_printed(text):
    assert I.find(text).cause == I.DISK_FULL


@pytest.mark.parametrize(
    "text, cause",
    [
        (_REPORT, "disk_full"),
        # The failed-install case: the detail need not carry the errno text.
        (
            "infrastructure_error: disk_full: Module file not found after the install step",
            "disk_full",
        ),
        # Behind the exception's own name, as the adapter's error reads.
        ("_MCPToolExecutionError: " + _REPORT, "disk_full"),
        # A sweep's fit pool, lost mid-fit, or refused at startup in the server's traceback
        # (PEtab-MCP#226): the pool is to blame, not the run.
        (
            "infrastructure_error: fit_pool: the fit's client lost the pool's scheduler: "
            "FuturesCancelledError('scheduler-connection-lost')",
            "fit_pool",
        ),
        (
            "petab_mcp.errors.InfrastructureError: infrastructure_error: fit_pool: the pool at "
            "tcp://cn081:8786 runs another environment (amici 1.1.0 there, 1.2.0 here): start a "
            "pool from this server's environment",
            "fit_pool",
        ),
        # A cause the server may add later is read, not assumed.
        ("infrastructure_error: out_of_memory: the worker was killed", "out_of_memory"),
    ],
)
def test_the_servers_report_names_its_cause(text, cause):
    hit = I.find(text)
    assert hit.cause == cause and hit.text.startswith("infrastructure_error: ")
    assert I.stop_reason(hit.cause) == f"infrastructure_error: {cause}"


def test_the_causes_are_named_as_the_server_names_them():
    """The two packages cannot share a constant (this module is stdlib only), so the names are
    restated here; a server that renamed one would leave a stop filed under a name nothing reads."""
    from petab_mcp import errors

    assert I.STOP_PREFIX == errors.INFRASTRUCTURE_ERROR
    assert (I.DISK_FULL, I.FIT_WEDGED, I.FIT_POOL) == (
        errors.DISK_FULL,
        errors.FIT_WEDGED,
        errors.FIT_POOL,
    )


@pytest.mark.parametrize(
    "text",
    [
        "[Errno 2] No such file or directory: 'model.xml'",
        # The disguise without its cause: what the 241 rejections said. Only the build log names
        # the disk, which is why LogWatch exists.
        "the model passed validation but could not be compiled/fit: Module file not found. "
        "This usually means it is ill-posed for estimation",
        "best nll 1,000,122.5 after 28 starts",
        "errno = 1228",
        # The run's own stop reason is a label, not a report (no detail after the cause).
        I.STOP_REASON,
    ],
)
def test_nothing_else_is(text):
    assert I.find(text) is None


def test_the_evidence_is_one_short_line_around_the_match():
    hit = I.find("x" * 5000 + "\n" + _LINK + "\n" + "y" * 5000)
    assert "Disk quota exceeded" in hit.text and "\n" not in hit.text and len(hit.text) < 400


def test_an_error_is_read_through_its_group_and_its_cause():
    # The anyio task group around the MCP session wraps the real error beside a cancellation.
    try:
        try:
            raise OSError(122, "Disk quota exceeded")
        except OSError as exc:
            raise RuntimeError("could not store the result") from exc
    except RuntimeError as exc:
        wrapped = BaseExceptionGroup(
            "unhandled errors in a TaskGroup", [exc, asyncio.CancelledError()]
        )
    assert I.find_in_error(wrapped).cause == I.DISK_FULL
    assert I.find_in_error(RuntimeError("could not store the result")) is None


def test_the_log_watch_reads_each_whole_line_once(tmp_path):
    log = tmp_path / "amici_build.log"
    log.write_text("[ 50%] Building CXX object\n")
    watch = I.LogWatch(log, tmp_path / "server.log")  # no server.log yet: reads as nothing
    assert watch() is None

    with log.open("a") as fh:
        fh.write(_LINK + "\ncollect2: error: ld returned 1 exit status\n/usr/bin/ld: final link")
    first = watch()
    assert first.cause == I.DISK_FULL
    assert first.text.startswith("amici_build.log: ") and "Disk quota exceeded" in first.text
    assert watch() is None and watch.lines == 1  # read once

    # The last line has no newline yet: it is read once complete, or when the server is gone.
    with log.open("a") as fh:
        fh.write(" failed: Disk quota exceeded")
    assert watch() is None
    assert watch(final=True) is not None
    assert watch.lines == 2 and watch.evidence == first


# --- the tool wrapper ------------------------------------------------------------------------


class _Tool:
    def __init__(self, name, coro):
        self.name, self.coroutine = name, coro


def _text(text):
    return [{"type": "text", "text": text}]


def test_a_result_on_a_full_disk_stops_the_run_and_nothing_more_reaches_the_server():
    """The 40 raw-errno results came back as REJECTIONS, which the wrapper never treats as
    errors; the model then re-sent them until the dedup guard refused."""
    calls = []

    async def _make_changes(**kw):
        calls.append("make_changes")
        return (_text(json.dumps({"applied": False, "error": _FIT_CRASH})), None)

    async def _get_model(**kw):
        calls.append("get_model")
        return (_text('{"ok": true}'), None)

    state = B._new_breaker_state()
    tools = [_Tool("make_changes", _make_changes), _Tool("get_model", _get_model)]
    B._wrap_tools_with_circuit_breaker(tools, state)

    with pytest.raises(I.InfrastructureError, match="Disk quota exceeded"):
        asyncio.run(tools[0].coroutine())
    with pytest.raises(I.InfrastructureError):  # refused before the server, not forwarded
        asyncio.run(tools[1].coroutine())

    assert calls == ["make_changes"]
    assert state["infrastructure"]["tool_calls"] == 1
    assert state["infrastructure"]["evidence"].text.startswith("make_changes: ")
    assert B.call_outcomes(state, "make_changes")["by_outcome"] == {"infrastructure_error": 1}


def test_a_raised_error_on_a_full_disk_stops_the_run_too():
    """`make_changes` is never disabled and its errors are RETURNED to the model; a full disk
    must not take that path, which hands the model something it cannot fix."""
    original = RuntimeError(_TOOL_ERROR)

    async def _make_changes(**kw):
        raise original

    state = B._new_breaker_state()
    tool = _Tool("make_changes", _make_changes)
    B._wrap_tools_with_circuit_breaker([tool], state)

    with pytest.raises(I.InfrastructureError) as raised:
        asyncio.run(tool.coroutine())
    assert raised.value.__cause__ is original
    assert state["infrastructure"]["tool_calls"] == 1


def test_the_servers_report_stops_the_run_instead_of_reaching_the_model():
    """PEtab-MCP#95 ends the call as an `is_error` result. The bridge raises it as a
    ToolException, and its `handle_tool_error` would hand the model that text as the tool's
    result; the wrapper sees it first. Built through the bridge's own conversion
    (`langchain.mcp`), so the exception is the one a real call raises."""
    pytest.importorskip("langchain.mcp")
    from langchain.mcp.tools import _convert_call_tool_result
    from mcp.types import CallToolResult, TextContent

    async def _make_changes(**kw):
        return _convert_call_tool_result(
            CallToolResult(content=[TextContent(type="text", text=_REPORT)], is_error=True)
        )

    state = B._new_breaker_state()
    tool = _Tool("make_changes", _make_changes)
    B._wrap_tools_with_circuit_breaker([tool], state)

    with pytest.raises(I.InfrastructureError) as raised:
        asyncio.run(tool.coroutine())
    assert raised.value.evidence.cause == I.DISK_FULL
    assert raised.value.evidence.text.startswith("make_changes: infrastructure_error: disk_full: ")


def test_an_external_tools_full_disk_is_not_this_runs():
    async def _lookup(**kw):
        return (_text("upstream cache: No space left on device"), None)

    state = B._new_breaker_state()
    tool = _Tool("get_reactome_info_by_identifier", _lookup)
    B._wrap_tools_with_circuit_breaker(
        [tool], state, transport_disable_names={"get_reactome_info_by_identifier"}
    )

    asyncio.run(tool.coroutine())
    assert state["infrastructure"]["tool_calls"] == 0


# --- the analysis ----------------------------------------------------------------------------


def test_the_stop_is_its_own_bucket_and_never_scored():
    # Work done against a full disk is not a partial result, so it is not `infra_partial`; and
    # not scoreable even with a Δ and a verdict, so a rule that keeps losses without a Δ in the
    # denominator (gemot#41) cannot count it either.
    row = {
        "stop_reason": I.STOP_REASON,
        "success": False,
        "delta_bic": 12.0,
        "head_version": 6,
        "n_fits": 7,
    }
    assert A.stop_class(row) == "infrastructure_error"
    assert "infrastructure_error" in A.STOP_CLASSES
    assert not A.is_scoreable(row)
    # The Phoenix experiment summary agrees: dropped, whatever it did first.
    assert experiments._is_infra_abort(I.STOP_REASON)
    assert experiments.is_noop_infra_abort({**row, "n_ops": 7})


# --- the runner ------------------------------------------------------------------------------


def _drive(monkeypatch, tmp_path, run_deepagents):
    """``runner.drive._drive`` on a zero-shot problem with the server, the clock and the judge
    stubbed; the judge must never be reached. Returns ``(exit_code, result)``."""
    from argparse import Namespace
    from types import SimpleNamespace

    import gemot.agent
    from gemot.runner import drive

    async def _call(tool, **_):
        return {"model_authored": False} if tool == "describe_problem" else {"species": []}

    async def _judge(*a, **kw):
        raise AssertionError("a run on a failed infrastructure must not be judged")

    clock = SimpleNamespace(calls=0, by_tool={}, reject_summary=lambda: {})
    monkeypatch.setattr(drive, "_wrap_call_timeout", lambda session: None)
    monkeypatch.setattr(drive, "_wrap_tool_clock", lambda session: clock)
    monkeypatch.setattr(drive, "_make_call", lambda session: _call)
    monkeypatch.setattr(drive, "_base_meta", lambda *a: {"run_started_at": "t0", "git_sha": "s"})
    monkeypatch.setattr(drive, "_write_live_marker", lambda *a: None)
    monkeypatch.setattr(drive, "_judge", _judge)
    monkeypatch.setattr(gemot.agent, "run_deepagents", run_deepagents)

    scenario = SimpleNamespace(
        name="toy",
        version="v1",
        system_prompt="sp",
        agent_task="t",
        agent_entities=[],
        example_dir=tmp_path,
        adequacy=None,
    )
    args = Namespace(
        driver="deepagents",
        phoenix=False,
        phoenix_keep_alive=False,
        api_key="sk-test",
        model="m",
        base_url="http://x",
        recursion_limit=10,
        temperature=1.0,
        reasoning_effort="default",
        n_starts=1,
        max_wall_seconds=None,
    )
    session = SimpleNamespace(call_tool=None)
    return asyncio.run(drive._drive(scenario, session, tmp_path, args))


def test_a_run_whose_build_log_shows_a_full_disk_is_not_judged(monkeypatch, tmp_path):
    """The driver returned as if nothing happened -- the build only said "ill-posed" -- but
    amici_build.log names the disk. The run is written unjudged, with the infrastructure label and
    the count, rather than scored against a server that could not build."""

    async def _run_deepagents(session, **kw):
        (tmp_path / "amici_build.log").write_text(_LINK + "\n")
        return {
            "stop_reason": "completed",
            "infrastructure_tool_calls": 0,
            "infrastructure_evidence": None,
        }

    exit_code, result = _drive(monkeypatch, tmp_path, _run_deepagents)

    assert exit_code == 1
    assert result["stop_reason"] == I.STOP_REASON
    assert result["infrastructure_log_lines"] == 1
    assert "amici_build.log" in result["infrastructure_evidence"]
    assert result["success"] is False and result["delta_bic"] is None
    assert json.loads((tmp_path / "result.json").read_text())["stop_reason"] == I.STOP_REASON


def test_an_infrastructure_error_escaping_the_driver_keeps_its_label_and_evidence(
    monkeypatch, tmp_path
):
    """The anyio teardown can carry the wrapper's error out of the driver inside a group, past the
    loop's own handling. The degraded result is written from the run's base meta, and must still
    say why it stopped, with the evidence and the log count."""
    evidence = I.Evidence(I.DISK_FULL, "make_changes: " + _REPORT)

    async def _run_deepagents(session, **kw):
        (tmp_path / "server.log").write_text("OSError: [Errno 122] Disk quota exceeded\n")
        raise BaseExceptionGroup(
            "unhandled errors in a TaskGroup",
            [I.InfrastructureError(evidence), asyncio.CancelledError()],
        )

    exit_code, result = _drive(monkeypatch, tmp_path, _run_deepagents)

    assert exit_code == 1
    assert result["stop_reason"] == I.STOP_REASON
    assert result["infrastructure_evidence"] == evidence.text
    assert result["infrastructure_log_lines"] == 1


def test_the_runners_own_calls_raise_on_the_servers_report():
    """The judge and preflight read through `_make_call`, which unwraps an `is_error` result to its
    text. The server's report must end the run there, not reach the judge as a string."""
    from types import SimpleNamespace

    from gemot.runner.clock import _make_call

    def _result(text):
        return SimpleNamespace(
            is_error=True, structured_content=None, content=[SimpleNamespace(text=text)]
        )

    class _Session:
        def __init__(self, text):
            self.text = text

        async def call_tool(self, tool, args):
            return _result(self.text)

    with pytest.raises(I.InfrastructureError) as raised:
        asyncio.run(_make_call(_Session(_REPORT))("get_fit_summary"))
    assert raised.value.evidence.text.startswith("get_fit_summary: infrastructure_error: ")

    # Any other error result is returned as before, for the caller to handle.
    other = "unknown version 7"
    assert asyncio.run(_make_call(_Session(other))("get_model", version=7)) == other
