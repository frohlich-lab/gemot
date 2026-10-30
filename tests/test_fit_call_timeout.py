"""A fit that outruns its call timeout is the lead's to hear about, once; a second is a wedge.

gemot#56: in the INDRA sweep a 150-start fit hung on the server (PEtab-MCP#108). Its call timed out
after 7560 s, and each of three resumes re-sent the identical fit into the server still holding the
first, waiting out the full timeout again: 8.4 h of the run's last 9 h. These pin the tool
wrapper's half: the first timeout goes back to the lead as the tool's result, and a further timeout
behind it, or the server saying that fit is still running, ends the run as an infrastructure stop.
Every other timeout keeps the resume it had.

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

        async def _coro(runtime=None, **kw):
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
    return asyncio.run(tool.coroutine(runtime=None, **kw))


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
def test_the_server_saying_the_fit_still_runs_stops_the_run_at_once(reply):
    """PEtab-MCP#108: the next make_changes answers at once instead of queueing behind the fit,
    so the run need not wait out a second timeout to know."""
    fit = _Tool("make_changes", _timeout(), reply)
    _wrap(fit)
    _call(fit)

    with pytest.raises(I.InfrastructureError) as raised:
        _call(fit)
    assert raised.value.evidence.cause == I.FIT_WEDGED
    assert "fit_in_progress" in raised.value.evidence.text


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
