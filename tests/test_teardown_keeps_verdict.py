"""A failure closing the server after the judge must not replace the verdict (#4).

Both runners judge from inside ``async with _petab_server(...)``, so an exception raised while
closing it (the ClientSession to a wedged server, an anyio task group unwinding) reaches the
runner's guard after ``result.json`` already holds the judged run. That guard used to overwrite
it with ``aborted:`` and no token usage, a row the aggregators drop as infrastructure noise; the
barebones runner's ``except (RuntimeError, TimeoutError)`` rewrote it as an unscoreable delivery.

The server, the judge and the agent are stand-ins: no model calls, AMICI builds or fits.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp", reason="harness-only dependency")

from gemot.runner import barebones as B  # noqa: E402
from gemot.runner import main as M  # noqa: E402
from gemot.runner.cli import build_parser  # noqa: E402
from gemot.scenario import Scenario  # noqa: E402

TOY = Path(__file__).parent / "data" / "toy_v1"

#: What the judge wrote: a scored run the aggregators count.
_JUDGED = {"success": True, "delta_bic": -12.5, "stop_reason": "completed"}

#: What the barebones agent returned; a degraded result must carry it.
_AGENT_OUT = {"stop_reason": "completed", "token_usage": {"total_tokens": 1234}}

#: A delivery :func:`.barebones.check_delivery` would pass to the judge.
_CLEAN_DELIVERY = {
    "design_modified": [],
    "shipped_rows_violated": [],
    "model_authored": True,
    "petab_lint": {"ok": True, "messages": []},
}

#: Raised while closing the server, built afresh per test so no traceback carries over. The
#: RuntimeError is the 2026-09-17 reproduction; the group is what anyio raises unwinding a task
#: group, and is a BaseException but not an Exception.
_TEARDOWN_FAILURES = [
    pytest.param(
        lambda: RuntimeError("Attempted to exit cancel scope in a different task"),
        id="RuntimeError",
    ),
    pytest.param(
        lambda: BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()]),
        id="group",
    ),
]


def _scenario(tmp_path: Path) -> Scenario:
    problem = tmp_path / "toy" / "problem"
    shutil.copytree(TOY, problem)
    return Scenario("toy", problem, lambda *args: None)


def _server(*, on_enter: BaseException | None = None, on_exit: BaseException | None = None):
    """A stand-in for ``_petab_server`` that fails starting up or closing down."""

    @asynccontextmanager
    async def server(work, build_root, args, *, handle_out=None, **kwargs):
        handle = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None), session=object())
        if handle_out is not None:
            handle_out["handle"] = handle
        if on_enter is not None:
            raise on_enter
        yield handle
        if on_exit is not None:
            raise on_exit

    return server


def _judge(work: Path) -> tuple[int, dict]:
    (work / "result.json").write_text(json.dumps(_JUDGED))
    return 0, dict(_JUDGED)


def _on_disk(scenario: Scenario) -> dict:
    return json.loads((scenario.example_dir / "work" / "result.json").read_text())


# --------------------------------------------------------------------------- main
def _run_main(tmp_path, monkeypatch, server) -> tuple[int, dict, dict]:
    scenario = _scenario(tmp_path)

    async def drive(scenario, session, work, args, **kwargs):
        return _judge(work)

    monkeypatch.setattr(M, "_petab_server", server)
    monkeypatch.setattr(M, "_drive", drive)
    monkeypatch.setattr(M, "archive_run", lambda *args: None)
    args = build_parser(scenario).parse_args(["--driver", "scripted"])
    code, result = asyncio.run(M._main_async(scenario, args))
    return code, result, _on_disk(scenario)


@pytest.mark.parametrize("failure", _TEARDOWN_FAILURES)
def test_main_keeps_the_judged_result_when_teardown_fails(tmp_path, monkeypatch, failure):
    code, result, on_disk = _run_main(tmp_path, monkeypatch, _server(on_exit=failure()))
    assert (code, result) == (0, _JUDGED)
    assert on_disk == _JUDGED


def test_main_still_degrades_a_failure_before_the_verdict(tmp_path, monkeypatch):
    """The guard's job is unchanged: a server that never came up still leaves a result.json."""
    failure = RuntimeError("server exited early (code -11)")
    code, result, on_disk = _run_main(tmp_path, monkeypatch, _server(on_enter=failure))
    assert code == 1
    assert result["stop_reason"] == "aborted: RuntimeError: server exited early (code -11)"
    assert on_disk == result


# --------------------------------------------------------------------------- barebones
def _run_barebones(
    tmp_path, monkeypatch, server, judge, check=lambda *args: _CLEAN_DELIVERY
) -> tuple[int, dict, dict]:
    scenario = _scenario(tmp_path)

    async def run_barebones(agent_dir, **kwargs):
        return dict(_AGENT_OUT)

    @asynccontextmanager
    async def no_bio_sessions(urls):
        yield []

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    # The pipeline imports the agent lazily; a module stand-in keeps deepagents out of the test.
    agent = SimpleNamespace(run_barebones=run_barebones)
    monkeypatch.setitem(sys.modules, "gemot.agent.barebones", agent)
    sandbox = {"bwrap": "-", "env_prefix": "-"}
    monkeypatch.setattr(B, "sandbox_preflight", lambda *args, **kwargs: sandbox)
    monkeypatch.setattr(B, "open_bio_sessions", no_bio_sessions)
    monkeypatch.setattr(B, "check_delivery", check)
    monkeypatch.setattr(B, "_petab_server", server)
    monkeypatch.setattr(B, "_judge_delivered", judge)
    monkeypatch.setattr(B, "archive_run", lambda *args: None)
    args = build_parser(scenario).parse_args(["--driver", "barebones"])
    # Tracing has no off switch on the command line; set up for real it registers a global tracer
    # and can launch a local Phoenix app, both of which outlive the test.
    args.phoenix = False
    code, result = asyncio.run(B._main_async_barebones(scenario, args))
    return code, result, _on_disk(scenario)


async def _judges(scenario, session, work, args, meta, t0):
    return _judge(work)


@pytest.mark.parametrize("failure", _TEARDOWN_FAILURES)
def test_barebones_keeps_the_judged_result_when_teardown_fails(tmp_path, monkeypatch, failure):
    """The RuntimeError used to come back as ``delivered_model_unscoreable``, the group as an
    ``aborted:`` result."""
    server = _server(on_exit=failure())
    code, result, on_disk = _run_barebones(tmp_path, monkeypatch, server, _judges)
    assert (code, result) == (0, _JUDGED)
    assert on_disk == _JUDGED


def test_barebones_blames_the_delivery_only_for_a_failed_start_up(tmp_path, monkeypatch):
    """A server that dies compiling or scoring the delivery is a loss on the delivery."""
    server = _server(on_enter=RuntimeError("server exited early (code 1)"))
    code, result, on_disk = _run_barebones(tmp_path, monkeypatch, server, _judges)
    assert code == 1
    assert result["unscoreable_reason"].startswith(
        "delivered_model_unscoreable: RuntimeError: server exited early (code 1)"
    )
    assert on_disk == result


async def _judge_fails(*args):
    raise RuntimeError("it broke")


def _check_fails(*args):
    raise RuntimeError("it broke")


@pytest.mark.parametrize(
    "fails",
    [
        pytest.param({"judge": _judge_fails}, id="judge"),
        pytest.param({"check": _check_fails}, id="delivery check"),
    ],
)
def test_a_late_failure_is_an_abort_that_keeps_the_agents_work(tmp_path, monkeypatch, fails):
    """A failure in the judge is not the delivery's, so not ``unscoreable``. Either way the
    degraded result carries what the agent returned, or the aggregation reads the run as zero-work
    noise and drops it; the delivery check runs before ``meta`` takes the agent's output in."""
    stages = {"judge": _judges, **fails}
    code, result, on_disk = _run_barebones(tmp_path, monkeypatch, _server(), **stages)
    assert code == 1
    assert result["stop_reason"] == "aborted: RuntimeError: it broke"
    assert "unscoreable_reason" not in result
    assert result["token_usage"] == _AGENT_OUT["token_usage"]
    assert on_disk == result
