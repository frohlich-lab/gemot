"""The stall watchdog's floor is sized from the ``n_starts`` the pending fit actually asked for.

``n_starts`` is a PER-CALL argument — ``make_changes``/``fit`` take one that replaces the run
default, the agent picks it at runtime, and ``scenario.build_true_model`` floors it higher still
(bruno 400, fujita 300). The stall watchdog was once sized to ``fit_time_budget(2 * run_default)``:
a guess that still sat below those floors while doubling how long a wedged non-fit step went
undetected. These tests pin where it now reads the real count, and the layering around it. A tool
call itself is no longer sized from its fit: it lives on the server's heartbeats
(tests/test_heartbeat_timeout.py, gemot#106).
"""

from __future__ import annotations

import pytest

from gemot import fit_budget as B
from gemot.agent.driver import _pending_fit_starts
from gemot.fit_budget import (
    FIT_BUDGET_CEILING_S,
    fit_time_budget,
    fit_time_budget_for,
)


DEFAULT = 50


# -- the stall watchdog reads the PENDING fit ------------------------------------------------
class _Msg:
    """Stand-in for a langchain message.

    `content` is set because EVERY real message has it, and `_pending_fit_starts` uses exactly
    that to tell an ordinary message with no tool calls (keep scanning back) from an object it
    does not recognise (raise, and record `unreadable_graph_state`). A double without it is not
    a message — it would make these tests assert against a shape that never occurs.
    """

    type = "ai"

    def __init__(self, tool_calls, content=""):
        self.tool_calls = tool_calls
        self.content = content


def _state(*tool_calls):
    return {"messages": [_Msg(list(tool_calls))]}


def test_a_pending_make_changes_arms_the_watchdog_for_its_own_fit():
    """stream_mode="values" emits the AIMessage one super-step BEFORE the tool node runs the
    fit, so the count is visible in time to size the very wait the fit happens in."""
    state = _state({"name": "make_changes", "args": {"n_starts": 300}})
    assert _pending_fit_starts(state, DEFAULT) == 300


def test_batched_fits_are_summed_not_maxed():
    """ToolNode runs a message's tool calls concurrently against ONE pyPESTO worker pool, so two
    fits in a batch contend rather than overlap for free — the step's wall clock is the summed
    wave count, and max() would fire the watchdog mid-batch on a healthy run."""
    state = _state(
        {"name": "make_changes", "args": {"n_starts": 120}},
        {"name": "make_changes", "args": {"n_starts": 400}},
    )
    assert _pending_fit_starts(state, DEFAULT) == 520


def test_a_non_fit_tool_call_leaves_no_fit_pending():
    """The whole point of sizing per step: a wedged read-only step must not be held open for the
    length of a fit it was never going to run — even if it happens to carry an n_starts key."""
    state = _state({"name": "get_data_regimes", "args": {"n_starts": 400}})
    assert _pending_fit_starts(state, DEFAULT) is None


def test_a_completed_fit_disarms_the_window():
    """Once the tool node has run, the last message is a ToolMessage with no tool_calls — the fit
    is over, and re-arming on it would keep the wide window forever."""
    assert _pending_fit_starts({"messages": [_Msg([])]}, DEFAULT) is None


def test_the_watchdog_reads_a_string_typed_n_starts_too():
    state = _state({"name": "make_changes", "args": {"n_starts": "300"}})
    assert _pending_fit_starts(state, DEFAULT) == 300


def test_asking_for_fewer_starts_does_not_shrink_the_watchdog_window_either():
    """Mirrors the transport rule — the floor lives in fit_time_budget_for, so both agree."""
    state = _state({"name": "make_changes", "args": {"n_starts": 5}})
    assert fit_time_budget_for(_pending_fit_starts(state, DEFAULT), DEFAULT) == fit_time_budget(
        DEFAULT
    )


def test_one_unparseable_argument_does_not_discard_a_siblings_real_count():
    """Coercion is per call: a junk n_starts on one tool call must not drop the window back to
    the default while a real 400-start fit is pending alongside it."""
    state = _state(
        {"name": "make_changes", "args": {"n_starts": "many"}},
        {"name": "make_changes", "args": {"n_starts": 400}},
    )
    assert _pending_fit_starts(state, DEFAULT) == 400


def test_an_unrecognised_state_degrades_rather_than_raising():
    """A langgraph/deepagents version bump must cost precision, not the run."""
    for state in (
        None,
        {},
        {"messages": []},
        {"messages": [object()]},
        "nonsense",
        _state({"name": "make_changes", "args": {"n_starts": "many"}}),
        _state({"args": {"n_starts": 400}}),
    ):
        assert _pending_fit_starts(state, DEFAULT) is None


def test_a_serialized_message_is_still_read():
    """langgraph may hand back messages as plain dicts rather than AIMessage objects; that must
    read as a pending fit, not as an unrecognised state that silently disarms the window."""
    state = {"messages": [{"tool_calls": [{"name": "make_changes", "args": {"n_starts": 300}}]}]}
    assert _pending_fit_starts(state, DEFAULT) == 300


# -- the window may not be unbounded --------------------------------------------------------
def test_a_hallucinated_n_starts_cannot_buy_an_unbounded_window():
    """`n_starts` is chosen by the MODEL and the server accepts any int >= 1, so an unclamped
    budget would compute a ~115-day timeout and switch the watchdog off for that step."""
    capped = fit_time_budget_for(10**9, DEFAULT)
    assert capped <= FIT_BUDGET_CEILING_S
    assert capped < fit_time_budget(10**9)  # the clamp actually bit


def test_the_fit_tool_set_is_derived_from_the_leads_actual_roster():
    """A hand-written set stops matching on the next rename (`set_model`/`change_parameters` ->
    `make_changes` already happened) and nothing would fail — every fit would quietly drop to the
    run-default window. It must intersect the real roster, and must not be empty."""
    from gemot.agent.driver import _FIT_TOOLS
    from gemot.agent.tools import ORCHESTRATOR_TOOLS

    assert _FIT_TOOLS, "no fitting tool matched the lead's roster — the watchdog is disarmed"
    assert _FIT_TOOLS <= set(ORCHESTRATOR_TOOLS)


# -- the stall diagnostic must not misreport the commonest case -------------------------------
def test_a_fit_that_omits_n_starts_still_reports_as_pending():
    """The server's docstring tells the model to omit n_starts unless it wants MORE than the
    default, so this is the COMMONEST wedged-fit case. Returning the bare default here (rather
    than None) is what stops the stall message reading "no fit pending" while a full multistart
    is in flight — the post-mortem would go to the provider layer instead of the fit."""
    state = _state({"name": "make_changes", "args": {"antimony": "x"}})
    assert _pending_fit_starts(state, DEFAULT) == DEFAULT


def test_no_fit_pending_is_distinguishable_from_a_default_sized_fit():
    """Both size the same window, but they are different diagnoses."""
    assert _pending_fit_starts(_state(), DEFAULT) is None
    assert _pending_fit_starts(_state({"name": "make_changes", "args": {}}), DEFAULT) == DEFAULT


# -- regressions the second review surfaced ---------------------------------------------------
def test_a_json_string_args_container_does_not_discard_a_valid_sibling():
    """`args` arriving as a JSON string (a real provider/serialization variant) makes
    `args.get` raise AttributeError. If that escapes the per-call guard it takes the WHOLE
    emission with it, so a genuine 400-start fit pending alongside collapses to the default."""
    state = _state(
        {"name": "make_changes", "args": '{"n_starts": 400}'},
        {"name": "make_changes", "args": {"n_starts": 400}},
    )
    assert _pending_fit_starts(state, DEFAULT) == 400


def test_a_negative_n_starts_never_reaches_the_stall_message():
    """The server rejects it, but the AIMessage is read one super-step earlier — without a clamp
    the post-mortem line reads "-5-start fit pending"."""
    state = _state({"name": "make_changes", "args": {"n_starts": -5}})
    assert _pending_fit_starts(state, DEFAULT) == 1


def test_an_unreadable_state_is_counted_where_a_sweep_can_see_it():
    """Falling back to the run-default window IS the old bug (bachmann task 9). It must not be
    absorbed silently: guard_stats is spread into result.json."""
    stats: dict = {}
    assert _pending_fit_starts("not-a-mapping", DEFAULT, stats) is None
    assert _pending_fit_starts(object(), DEFAULT, stats) is None
    assert stats["unreadable_graph_state"] == 2


def test_a_healthy_state_records_nothing():
    stats: dict = {}
    _pending_fit_starts(_state({"name": "make_changes", "args": {"n_starts": 300}}), DEFAULT, stats)
    _pending_fit_starts(_state(), DEFAULT, stats)
    assert "unreadable_graph_state" not in stats


def test_a_per_call_timeout_is_told_apart_from_an_ordinary_tool_error():
    """mcp turns an expired read_timeout_seconds into MCPError(REQUEST_TIMEOUT), which is not in
    the httpx2 family — so without this it reads as an ordinary tool error and goes back to the model with
    nothing to say the server is still running the abandoned multistart. A fit's goes to the lead
    as a note that says so, and any other call's resumes, as the session-wide sse_read_timeout's
    httpx.ReadTimeout did (tests/test_fit_call_timeout.py)."""
    from mcp import MCPError
    from mcp.types import REQUEST_TIMEOUT

    from gemot.agent.circuit_breaker import _is_call_timeout, _is_transport_error

    timed_out = MCPError(code=REQUEST_TIMEOUT, message="Timed out waiting for response")
    assert _is_call_timeout(timed_out)
    assert _is_transport_error(timed_out)
    # A 400 is a deterministic tool error and MUST still reach the model.
    bad_request = MCPError(code=400, message="invalid antimony")
    assert not _is_call_timeout(bad_request)
    assert not _is_transport_error(bad_request)
    assert not _is_transport_error(ValueError("ordinary failure"))


# -- the third review's findings ---------------------------------------------------------------
def test_an_absurd_n_starts_is_clamped_not_overflowed():
    """The seconds ceiling is applied to the RESULT, so without a COUNT cap the formula still
    evaluates at the raw value first — and 10**400 overflows `waves * per_start` into an
    OverflowError that ends the run, defeating the guard entirely. Reachable: the count is read
    straight off the model's unvalidated tool-call JSON."""
    for absurd in (10**400, float("inf"), -(10**400)):
        assert fit_time_budget_for(absurd, DEFAULT) <= FIT_BUDGET_CEILING_S


def test_the_default_floor_survives_a_ceiling_that_would_cut_below_it():
    """The ceiling bounds what the MODEL may ask for, never what the operator configured. A run
    whose own default budget exceeds the ceiling must not be silently cut below it."""
    from gemot.fit_budget import _MAX_DERIVABLE_N_STARTS

    # Below the COUNT clamp, so the floor is this run's real default budget...
    huge = _MAX_DERIVABLE_N_STARTS // 10
    assert fit_time_budget(huge) > FIT_BUDGET_CEILING_S, "pick a default past the ceiling"
    assert fit_time_budget_for(None, huge) == fit_time_budget(huge)
    assert fit_time_budget_for(1, huge) == fit_time_budget(huge)
    # ...and the count clamp applies to the default too, or the overflow guard has a hole in it.
    assert fit_time_budget_for(None, 10**400) == fit_time_budget(_MAX_DERIVABLE_N_STARTS)


def test_the_three_timeouts_are_strictly_ordered_not_merely_all_large():
    """httpx ceiling > watchdog > a call's heartbeat cut, each measured from the last heartbeat.
    With the watchdog at or under the cut, a silent server's fit is cancelled mid-flight as
    'stalled' and the clean timeout that resumes without a re-fit is unreachable; with the httpx
    ceiling under it, the transport is torn down instead."""
    from gemot.agent.driver import _STEP_STALL_TIMEOUT_S
    from gemot.runner.clock import CALL_IDLE_WINDOW_S

    ceiling = max(FIT_BUDGET_CEILING_S, fit_time_budget_for(None, DEFAULT)) * 1.2
    assert CALL_IDLE_WINDOW_S < _STEP_STALL_TIMEOUT_S < ceiling


def test_a_pending_fit_survives_a_middleware_message_landing_after_it():
    """deepagents/langchain middleware hooks are graph NODES with their own emissions. Reading
    only messages[-1] meant any of them landing between the AIMessage and the tool node dropped
    the window to the floor — bachmann task 9 again, and invisible, because the state SHAPE is
    fine so `unreadable_graph_state` never fires."""
    ai = _Msg([{"name": "make_changes", "args": {"n_starts": 300}}])
    nudge = _Msg([])  # a guard/compaction emission
    assert _pending_fit_starts({"messages": [ai, nudge]}, DEFAULT) == 300


def test_the_scan_stops_at_a_tool_result_so_a_finished_fit_does_not_re_arm():
    """Once results are back the fit is over, however many messages follow it."""
    ai = _Msg([{"name": "make_changes", "args": {"n_starts": 300}}])
    result = _Msg([])
    result.type = "tool"  # a ToolMessage: results are back, the fit is over
    assert _pending_fit_starts({"messages": [ai, result]}, DEFAULT) is None
    assert _pending_fit_starts({"messages": [ai, {"type": "tool"}]}, DEFAULT) is None


def test_a_fit_timeout_abort_scores_as_infrastructure_not_as_a_failed_model():
    """It is the same event as the httpx ReadTimeout it replaced — it only stopped being one when
    the timeout moved from the session to the individual call. Unmarked, a 0-fit run killed by a
    fit timeout enters the success/dBIC aggregation as a genuine modelling failure."""
    from gemot.experiments import _is_infra_abort

    assert _is_infra_abort(
        {
            "stop_reason": "fit call timed out: Timed out while waiting for "
            "response to CallToolRequest. Waited 7560.0 seconds."
        }
    )
    assert not _is_infra_abort({"stop_reason": "completed"})


# -- a fit on a sweep's fit pool (gemot#117) -------------------------------------------------
def alkan(monkeypatch) -> None:
    """gemot#117's run: a 2,400 s per-start cap on a job of 8 CPUs, at the default margins. Also
    used by tests/test_run_loop_no_model.py, which replays the run's stop."""
    for name, value in (
        ("FIT_PER_START_TIMEOUT_S", 2400.0),
        ("FIT_N_WORKERS", 8),
        ("FIT_COMPILE_MARGIN_S", 600.0),
        ("FIT_BUDGET_MARGIN", 1.4),
    ):
        monkeypatch.setattr(B, name, value)


def test_a_fit_on_the_pool_is_one_wave_and_the_timeouts_keep_the_jobs_waves(monkeypatch):
    """alkan's 100 starts on the job's 8 CPUs are 13 waves: the 44,520 s of headroom its log
    named when it stopped at 15.4 h of 27.8 h. On the pool's 384 workers they run at once, and
    its fits took 2,400-3,000 s. One wave (4,200 s) covers that; the 13 waves stay on every
    timeout, which a live heartbeat moves out anyway."""
    alkan(monkeypatch)
    assert B.fit_time_budget(100) == pytest.approx(44_520)
    assert B.fit_time_budget(100, pool=True) == pytest.approx(4_200)
    assert B.fit_time_budget_for(None, 100) == pytest.approx(44_520)
