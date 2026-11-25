"""A fit that outruns its call timeout is the lead's to hear about, once; a silent server behind
it is a wedge.

gemot#56: in the INDRA sweep a 150-start fit hung on the server (PEtab-MCP#108). Its call timed out
after 7560 s, and each of three resumes re-sent the identical fit into the server still holding the
first, waiting out the full timeout again: 8.4 h of the run's last 9 h. These pin the tool
wrapper's half: the first timeout goes back to the lead as the tool's result, and a further timeout
behind it ends the run as an infrastructure stop. Every other timeout keeps the resume it had.

gemot#160: a server that answers is not wedged. On luna froehlich (63982610_55) the first compile ran
past its margin, the lead resent the change, and the server's `fit_in_progress` ended the run with
no fit. A call cut while the server's heartbeats show its change compiling, and `fit_in_progress`,
now reach their caller. The replay over a real session is in tests/test_heartbeat_timeout.py.

The driver loop's half is in tests/test_run_loop_no_model.py, next to the other loop exits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from gemot import analysis as A
from gemot import experiments
from gemot import infrastructure as I
from gemot import prompts
from gemot.agent import circuit_breaker as B

pytest.importorskip("mcp", reason="harness-only dependency")

from mcp import MCPError  # noqa: E402
from mcp.types import REQUEST_TIMEOUT  # noqa: E402

# What mcp raised in the sweep (logs/sw55big.61134771_0.out).
_TIMEOUT = "Timed out while waiting for response to ClientRequest. Waited 7560.0 seconds."
_STILL_RUNNING = "fit_in_progress: a 150-start fit is still running on this session"


def _timeout():
    return MCPError(code=REQUEST_TIMEOUT, message=_TIMEOUT)


def _text(text):
    return [{"type": "text", "text": text}]


class _Tool:
    """A tool whose coroutine plays ``answers`` in order: an exception is raised, anything else
    returned. Records each call that reached it."""

    def __init__(self, name, *answers):
        self.name, self.answers, self.calls = name, list(answers), 0

        async def _coro(**kw):
            self.calls += 1
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        self.coroutine = _coro


def _wrap(*tools):
    state = B._new_breaker_state()
    B._wrap_tools_with_circuit_breaker(list(tools), state, fit_tools=frozenset({"make_changes"}))
    return state


def _call(tool, **kw):
    return asyncio.run(tool.coroutine(**kw))


_FIT_OK = (_text(json.dumps({"applied": True, "version": 3})), None)


def test_a_fit_that_outruns_its_timeout_reaches_the_lead_as_its_result():
    """Not a resume: the resume re-sent the same fit into the server still running it."""
    fit = _Tool("make_changes", _timeout())
    state = _wrap(fit)

    content, artifact = _call(fit, n_starts=150)

    assert content == prompts.FIT_CALL_TIMEOUT_TMPL.format(
        name="make_changes", error=f"MCPError: {_TIMEOUT}"
    )
    assert artifact is None
    assert state["fit_timeout"] == f"make_changes: MCPError: {_TIMEOUT}"
    assert B.call_outcomes(state, "make_changes")["by_outcome"] == {"tool_error": 1}


@pytest.mark.parametrize(
    "behind",
    [
        "make_changes",  # the lead's next fit: the sweep's resend, once
        "get_model",  # a read queued on the same session lock
    ],
)
def test_a_further_timeout_behind_it_stops_the_run_as_a_wedged_fit(behind):
    fit = _Tool("make_changes", _timeout(), _timeout())
    read = _Tool("get_model", _timeout())
    later = _Tool("list_versions", (_text("{}"), None))
    state = _wrap(fit, read, later)
    _call(fit)

    with pytest.raises(I.InfrastructureError) as raised:
        _call(fit if behind == "make_changes" else read)
    evidence = raised.value.evidence
    assert evidence.cause == I.FIT_WEDGED
    assert evidence.text == (
        f"{behind}: MCPError: {_TIMEOUT}; behind make_changes: MCPError: {_TIMEOUT}"
    )
    assert state["infrastructure"]["evidence"] == evidence
    # The rest of the run queues behind the same fit, so it is refused before the server.
    with pytest.raises(I.InfrastructureError):
        _call(later)
    assert later.calls == 0


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError(f"Error calling tool 'make_changes': {_STILL_RUNNING}"),  # an is_error
        (_text(json.dumps({"applied": False, "error": _STILL_RUNNING})), None),  # a rejection
    ],
    ids=["raised", "returned"],
)
def test_the_server_saying_the_fit_still_runs_reaches_the_lead(reply):
    """gemot#160: the server answering is a server working, and its reply says to call
    `wait_for_fit`. It used to end the run as `fit_wedged`, with the lead never seeing it. The fit
    is still outstanding, so the server going silent behind it is still a wedge."""
    fit = _Tool("make_changes", _timeout(), reply, _timeout())
    state = _wrap(fit)
    _call(fit)
    held = state["fit_timeout"]

    content, _ = _call(fit)
    assert _STILL_RUNNING in (content if isinstance(content, str) else content[0]["text"])
    assert state["infrastructure"]["tool_calls"] == 0
    assert state["fit_timeout"] == held
    with pytest.raises(I.InfrastructureError) as raised:
        _call(fit)
    assert raised.value.evidence.cause == I.FIT_WEDGED


# What `runner.clock._wrap_call_timeout` raises on a call whose heartbeats show a compile past its
# margin: 63982610_55's, at 877 s.
_COMPILING = "compiling for 877s"


def _compile_cut(name="make_changes"):
    return MCPError(
        code=REQUEST_TIMEOUT,
        message=f"Request {name!r} timed out: its change has been {_COMPILING} (the compile "
        "margin is 600s)",
        data={"heartbeat": _COMPILING},
    )


def test_a_fit_cut_on_a_slow_compile_tells_the_lead_it_runs_on():
    """gemot#160: the lead was told the result was unknown and to choose fewer starts, so it
    resent the change. The server was compiling it, and fits it once built."""
    fit = _Tool("make_changes", _compile_cut())
    state = _wrap(fit)

    content, artifact = _call(fit)

    assert content == prompts.FIT_STILL_COMPILING_TMPL.format(
        name="make_changes", heartbeat=_COMPILING
    )
    assert "wait_for_fit" in content and "fewer starts" not in content
    assert artifact is None
    assert state["fit_timeout"] is not None


@pytest.mark.parametrize("behind", ["make_changes", "get_model_structure", "wait_for_fit"])
def test_a_cut_on_a_slow_compile_behind_a_timed_out_fit_is_no_wedge(behind):
    """gemot#160, 63782292_55: after the fit's cut, the lead's `get_model_structure` waited behind
    the same compile and was cut too, which ended the run as `fit_wedged`. A server still beating
    is working, so each cut reaches its caller; only silence behind it is a wedge."""
    fit = _Tool("make_changes", _compile_cut(), _compile_cut())
    other = _Tool(behind, _compile_cut(behind))
    state = _wrap(fit, other)
    _call(fit)

    content, _ = _call(fit if behind == "make_changes" else other)

    if behind == "make_changes":
        assert content.startswith("FIT STILL RUNNING")
    else:
        assert content == prompts.CALL_BEHIND_COMPILE_TMPL.format(name=behind, heartbeat=_COMPILING)
    assert state["infrastructure"]["tool_calls"] == 0


@pytest.mark.parametrize(
    "status, ended", [("committed", True), ("no_version", True), ("still_running", False)]
)
def test_wait_for_fit_saying_the_fit_ended_ends_the_wait(status, ended):
    """The lead's wait on the fit it was told of: once the server says no change is running, a
    later timeout is a new one, the lead's to hear about, not a wedge."""
    fit = _Tool("make_changes", _timeout(), _timeout())
    wait = _Tool("wait_for_fit", (_text(json.dumps({"status": status})), None))
    state = _wrap(fit, wait)
    _call(fit)

    _call(wait)

    assert (state["fit_timeout"] is None) is ended
    if ended:
        content, _ = _call(fit)
        assert content.startswith("FIT TIMED OUT")
    else:
        with pytest.raises(I.InfrastructureError):
            _call(fit)


def test_a_fit_that_returns_ends_the_wait_so_a_later_timeout_reaches_the_lead_again():
    """A slow fit is not a wedged one: once the server answers a fit, the one that timed out has
    ended, and the next timeout is the lead's to hear about too."""
    fit = _Tool("make_changes", _timeout(), _FIT_OK, _timeout())
    state = _wrap(fit)

    _call(fit)
    assert _call(fit) == _FIT_OK
    assert state["fit_timeout"] is None
    content, _ = _call(fit)
    assert content.startswith("FIT TIMED OUT")


def test_a_read_answered_meanwhile_does_not_end_the_wait():
    """A read the server answers beside a running fit says nothing about that fit."""
    fit = _Tool("make_changes", _timeout(), _timeout())
    read = _Tool("get_model", (_text("{}"), None))
    _wrap(fit, read)

    _call(fit)
    _call(read)
    with pytest.raises(I.InfrastructureError):
        _call(fit)


def test_still_running_with_no_fit_timed_out_is_the_servers_answer_to_the_lead():
    """Without a timed-out fit, the fit still running is one batched beside this call."""
    fit = _Tool(
        "make_changes", (_text(json.dumps({"applied": False, "error": _STILL_RUNNING})), None)
    )
    state = _wrap(fit)

    content, _ = _call(fit)
    assert _STILL_RUNNING in content[0]["text"]
    assert state["infrastructure"]["tool_calls"] == 0


def test_a_reads_timeout_with_no_fit_timed_out_still_takes_the_resume():
    """Unchanged: a call timeout that is not a fit's still raises, for the driver to resume."""
    read = _Tool("get_model", _timeout())
    state = _wrap(read)

    with pytest.raises(MCPError):
        _call(read)
    assert state["fit_timeout"] is None


def test_a_wedged_fit_is_an_infrastructure_stop_kept_out_of_every_success_rate():
    """Its own label, not judged, not scoreable, and dropped from the Phoenix summary as the
    sweep resubmits it, like a full disk (gemot#39)."""
    row = {
        "stop_reason": I.stop_reason(I.FIT_WEDGED),
        "success": False,
        "delta_bic": 3.0,
        "head_version": 12,
        "n_fits": 12,
    }
    assert row["stop_reason"] == "infrastructure_error: fit_wedged"
    assert A.stop_class(row) == "infrastructure_error"
    assert not A.is_scoreable(row)
    assert experiments.is_noop_infra_abort({**row, "n_ops": 13})
