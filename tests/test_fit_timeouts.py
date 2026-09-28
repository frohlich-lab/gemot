"""Every fit timeout is sized from the ``n_starts`` that fit actually asked for.

``n_starts`` is a PER-CALL argument — ``make_changes``/``fit`` take one that replaces the run
default, the agent picks it at runtime, and ``scenario.build_true_model`` floors it higher still
(bruno 400, fujita 300). Both the transport timeout and the stall watchdog were once sized to
``fit_time_budget(2 * run_default)``: a guess that still sat below those floors while doubling how
long a wedged non-fit step went undetected. These tests pin the two places that now read the real
count, and the layering that makes it work.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from petab_harness.agent.driver import _pending_fit_starts
from petab_harness.fit_budget import (
    FIT_BUDGET_CEILING_S,
    fit_time_budget,
    fit_time_budget_for,
)
from petab_harness.runner.clock import _make_call, _wrap_call_timeout


class _FakeSession:
    """Records the kwargs each call_tool reached the transport with."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {}

    def timeout_of(self, i=0):
        return self.calls[i][1].get("read_timeout_seconds")


DEFAULT = 50


def _wrapped():
    s = _FakeSession()
    _wrap_call_timeout(s, DEFAULT)
    return s


# -- per-call transport timeout -------------------------------------------------------------
def test_a_call_asking_for_more_starts_gets_that_fits_budget():
    """The motivating case: build_true_model's 400 starts against a run default of 50."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"antimony": "x", "n_starts": 400}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(400, DEFAULT))
    # ...and never LESS than the old `2 * default` guess. Not a strict `>`: fit_time_budget
    # counts worker WAVES, so on a box with FIT_N_WORKERS >= 400 both counts are one wave and
    # the budgets are legitimately equal. FIT_N_WORKERS is bound at import from the environment
    # and no test pins it, so a strict comparison would pass or fail by allocation size.
    assert s.timeout_of() >= timedelta(seconds=fit_time_budget_for(2 * DEFAULT, DEFAULT))


def test_keyword_arguments_are_read_too():
    """Both real callers (`_make_call` and langchain-mcp-adapters) pass name+arguments
    POSITIONALLY, with only progress_callback by keyword — the positional path is the one that
    carries production traffic. The keyword form is accepted for any other caller."""
    s = _wrapped()
    asyncio.run(s.call_tool(name="make_changes", arguments={"n_starts": 300}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(300, DEFAULT))


def test_a_tool_that_cannot_fit_gets_the_short_budget_not_the_run_default():
    """A read used to inherit the run-default FIT budget — 4200s at bachmann's settings — so a
    wedged one cost a whole multistart of wall clock and left no trace. It is sized on what the
    tool can do now, not on whether it happened to carry `n_starts`."""
    from petab_harness.runner.clock import NON_FIT_CALL_TIMEOUT_S

    s = _wrapped()
    asyncio.run(s.call_tool("get_fit_summary", {}))
    assert s.timeout_of() == timedelta(seconds=NON_FIT_CALL_TIMEOUT_S)


def test_an_edit_tool_that_omits_n_starts_still_gets_a_FIT_budget():
    """`make_changes` fits at the server's default when the agent omits the count, so "no
    n_starts" must not be read as "cannot fit" — that would cut a healthy fit at 900s."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(None, DEFAULT))


def test_asking_for_fewer_starts_does_not_shrink_the_window():
    """n_starts below the default makes a SHORTER fit, but the default budget is the floor we
    already trust and cutting below it buys nothing — it only risks killing a healthy call."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": 5}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(None, DEFAULT))


def test_an_explicit_caller_timeout_is_never_overridden():
    s = _wrapped()
    asyncio.run(
        s.call_tool("make_changes", {"n_starts": 400}, read_timeout_seconds=timedelta(seconds=7))
    )
    assert s.timeout_of() == timedelta(seconds=7)


def test_other_kwargs_are_passed_through_untouched():
    """langchain-mcp-adapters sends progress_callback= on every single call and relies on it
    arriving; a wrapper that rebuilt the kwargs would silently drop it."""
    sentinel = object()
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": 400}, progress_callback=sentinel))
    args, kwargs = s.calls[0]
    assert args == ("make_changes", {"n_starts": 400})
    assert kwargs["progress_callback"] is sentinel


def test_a_string_typed_n_starts_is_read_like_the_server_reads_it():
    """The server coerces with a bare `int()` (`session._coerce_n_starts`), so a model that sends
    "400" as a string still runs a 400-start fit. The timeout must agree about what was asked for
    or it silently under-sizes exactly the calls most likely to be long."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": "400"}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(400, DEFAULT))


def test_a_malformed_arguments_payload_still_gets_a_timeout():
    """A timeout must never be the thing that fails a call."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", "not-a-dict"))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(None, DEFAULT))


def test_the_scripted_path_gets_the_scenario_floor_without_Scenario_exposing_it():
    """`build_true_model` floors n_starts at min_n_starts, and the scripted driver has no agent
    loop — the per-call timeout is its ONLY guard. It must see 400, not the run's 50, even though
    `runner.main` (which sets the session timeout) cannot know the floor exists."""
    from petab_harness.scenario import build_true_model

    s = _wrapped()
    asyncio.run(
        build_true_model(
            _make_call(s),
            antimony="A -> B; k*A",
            justification="j",
            n_starts=DEFAULT,
            min_n_starts=400,
        )
    )
    assert s.calls[0][0][0] == "make_changes"
    # `fit_time_budget_for`, not `fit_time_budget`: the production path CLAMPS at
    # FIT_BUDGET_CEILING_S and the raw form does not, so comparing against the raw form only
    # holds on a machine big enough never to reach the cap. `FIT_N_WORKERS` defaults to the core
    # count, so 400 starts is 400 sequential waves on CI's 2-core runner (101640s, clamped to
    # 86400) and a handful of waves on an 8-core box (nowhere near it). That is why this passed
    # locally and failed in CI on py3.12 AND py3.13 -- the same machine-dependence the wave-count
    # comment at the top of this file already warns about.
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(400, DEFAULT))


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


# -- the two readers must agree, and neither may be unbounded ---------------------------------
def test_the_watchdog_and_the_transport_derive_the_SAME_budget():
    """They read n_starts from different places (a pending tool_calls entry vs a live call_tool
    argument) but must turn it into the same number of seconds — otherwise whichever is
    accidentally tighter silently becomes the real limit."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": 400}))
    state = _state({"name": "make_changes", "args": {"n_starts": 400}})
    watchdog = fit_time_budget_for(_pending_fit_starts(state, DEFAULT), DEFAULT)
    assert s.timeout_of() == timedelta(seconds=watchdog)


def test_a_hallucinated_n_starts_cannot_buy_an_unbounded_window():
    """`n_starts` is chosen by the MODEL and the server accepts any int >= 1, so an unclamped
    budget would compute a ~115-day timeout and switch both timeouts off for that call."""
    capped = fit_time_budget_for(10**9, DEFAULT)
    assert capped <= FIT_BUDGET_CEILING_S
    assert capped < fit_time_budget(10**9)  # the clamp actually bit
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": 10**9}))
    assert s.timeout_of() == timedelta(seconds=capped)


def test_a_positionally_supplied_timeout_does_not_break_the_call():
    """call_tool's third positional IS read_timeout_seconds; injecting the keyword on top of it
    raises "got multiple values for argument" — the wrapper failing the call it is protecting."""
    s = _wrapped()
    asyncio.run(s.call_tool("make_changes", {"n_starts": 400}, timedelta(seconds=9)))
    assert s.calls[0][0][2] == timedelta(seconds=9)
    assert "read_timeout_seconds" not in s.calls[0][1]


def test_the_fit_tool_set_is_derived_from_the_leads_actual_roster():
    """A hand-written set stops matching on the next rename (`set_model`/`change_parameters` ->
    `make_changes` already happened) and nothing would fail — every fit would quietly drop to the
    run-default window. It must intersect the real roster, and must not be empty."""
    from petab_harness.agent.driver import _FIT_TOOLS
    from petab_harness.agent.tools import ORCHESTRATOR_TOOLS

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


def test_the_transport_wrapper_is_deliberately_tool_agnostic():
    """Unlike the watchdog's `_FIT_TOOLS`, the transport cut keys off the PRESENCE of n_starts
    rather than the tool name. Nothing but `make_changes` takes one today, and keeping it
    name-free means a future fitting tool is covered the day it is added rather than the day
    someone remembers to add it to a list."""
    s = _wrapped()
    asyncio.run(s.call_tool("some_future_fitting_tool", {"n_starts": 400}))
    assert s.timeout_of() == timedelta(seconds=fit_time_budget_for(400, DEFAULT))


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


def test_a_per_call_fit_timeout_takes_the_RESUME_path_not_the_model():
    """mcp turns an expired read_timeout_seconds into McpError(408), which is not in the httpx
    family — so without this it reads as an ordinary tool error and goes back to the model, which
    re-fits while the server is still running the abandoned multistart on the same worker pool.
    The session-wide sse_read_timeout it replaced raised httpx.ReadTimeout and resumed."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    from petab_harness.agent.circuit_breaker import _is_call_timeout, _is_transport_error

    timed_out = McpError(ErrorData(code=408, message="Timed out waiting for response"))
    assert _is_call_timeout(timed_out)
    assert _is_transport_error(timed_out)
    # A 400 is a deterministic tool error and MUST still reach the model.
    bad_request = McpError(ErrorData(code=400, message="invalid antimony"))
    assert not _is_call_timeout(bad_request)
    assert not _is_transport_error(bad_request)
    assert not _is_transport_error(ValueError("ordinary failure"))


def test_the_session_ceiling_stays_strictly_above_every_per_call_budget():
    """If they were equal, the clamped worst case would be a race between the anyio timer and
    httpx's read deadline — and an httpx win tears down the transport instead of returning a
    clean 408."""
    from petab_harness.runner import main

    ceiling = max(FIT_BUDGET_CEILING_S, fit_time_budget_for(None, DEFAULT)) * 1.2
    assert ceiling > FIT_BUDGET_CEILING_S
    assert main.FIT_BUDGET_CEILING_S == FIT_BUDGET_CEILING_S
    assert fit_time_budget_for(10**9, DEFAULT) < ceiling


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
    from petab_harness.fit_budget import _MAX_DERIVABLE_N_STARTS

    # Below the COUNT clamp, so the floor is this run's real default budget...
    huge = _MAX_DERIVABLE_N_STARTS // 10
    assert fit_time_budget(huge) > FIT_BUDGET_CEILING_S, "pick a default past the ceiling"
    assert fit_time_budget_for(None, huge) == fit_time_budget(huge)
    assert fit_time_budget_for(1, huge) == fit_time_budget(huge)
    # ...and the count clamp applies to the default too, or the overflow guard has a hole in it.
    assert fit_time_budget_for(None, 10**400) == fit_time_budget(_MAX_DERIVABLE_N_STARTS)


def test_the_three_timeouts_are_strictly_ordered_not_merely_all_large():
    """httpx ceiling > watchdog > per-call cut. Ties are unsafe: with the watchdog equal to the
    transport cut it always wins (armed a graph-dispatch earlier), the step is cancelled
    mid-flight as 'stalled', and the clean 408 resume path is unreachable."""
    from petab_harness.fit_budget import FIT_WATCHDOG_MARGIN

    per_call = fit_time_budget_for(400, DEFAULT)
    watchdog = per_call * FIT_WATCHDOG_MARGIN
    ceiling = max(FIT_BUDGET_CEILING_S, fit_time_budget_for(None, DEFAULT)) * 1.2
    assert per_call < watchdog < ceiling


def test_a_second_batched_fit_is_sized_for_its_wait_on_the_server_lock():
    """`make_changes` holds session.lock across its whole change-and-fit critical section, so
    batched fits serialize. The second call waits out the first entirely before its own starts —
    sizing it at its own count alone 408s a healthy call at roughly half the time it needs."""
    import asyncio as _aio

    s = _FakeSession()
    gate = _aio.Event()

    async def slow(*args, **kwargs):
        s.calls.append((args, kwargs))
        await gate.wait()
        return {}

    s.call_tool = slow
    _wrap_call_timeout(s, DEFAULT)

    async def drive():
        first = _aio.create_task(s.call_tool("make_changes", {"n_starts": 400}))
        await _aio.sleep(0)  # let it arm and register in flight
        second = _aio.create_task(s.call_tool("make_changes", {"n_starts": 400}))
        await _aio.sleep(0)
        gate.set()
        await _aio.gather(first, second)

    _aio.run(drive())
    assert s.timeout_of(0) == timedelta(seconds=fit_time_budget_for(400, DEFAULT))
    assert s.timeout_of(1) == timedelta(seconds=fit_time_budget_for(800, DEFAULT))


def test_an_in_flight_fit_is_released_when_the_call_returns():
    """Otherwise every later fit inherits a phantom queue and the windows grow without bound."""
    s = _wrapped()
    for _ in range(3):
        asyncio.run(s.call_tool("make_changes", {"n_starts": 400}))
    assert s.timeout_of(2) == timedelta(seconds=fit_time_budget_for(400, DEFAULT))


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
    from petab_harness.experiments import _is_infra_abort

    assert _is_infra_abort(
        {
            "stop_reason": "fit call timed out: Timed out while waiting for "
            "response to CallToolRequest. Waited 7560.0 seconds."
        }
    )
    assert not _is_infra_abort({"stop_reason": "completed"})
