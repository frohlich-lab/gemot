"""The MCP call handle the pipeline uses, and the tool clock wrapped around it.

``_ToolClock`` measures the wall-clock the run actually spent INSIDE tool calls. It tracks the
UNION of busy intervals rather than their sum, so concurrent analyst calls are not double-counted
and the reported compute time stays comparable across runs with different panel concurrency.
"""

from __future__ import annotations

import contextlib
import json
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession

from .. import infrastructure
from ..causal_constraints import CONSTRAINT_VIOLATED, broken_constraints
from ..fit_budget import fit_time_budget_for
from ..paths import validation_measurements_path


# --------------------------------------------------------------------------- MCP glue
def _unwrap(result):
    """Turn a CallToolResult into the tool's Python return value."""
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        # FastMCP wraps non-dict returns as {"result": <value>}.
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    texts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:
        return joined


#: Per-call budget for a call that CANNOT fit -- every read, plus `restore_version`, which reuses
#: a stored fit without recompiling or refitting. Applied in `_wrap_call_timeout` (below) so it
#: reaches EVERY caller: the judge, the driver's probes, and the agent panel's own reads, all of
#: which used to inherit the whole fit budget because the sizing keyed only on `n_starts`.
#:
#: Deliberately NOT a per-caller override. Two earlier attempts bounded individual handles by
#: passing `read_timeout_seconds` explicitly, which sets `override` below and so DISCARDS the
#: `in_flight` queue term -- and the queue term is precisely what keeps a read honest when it is
#: sitting behind a live multistart. The finish gate reads through such a handle from INSIDE a
#: super-step (`_finish_tool(..., probe=progress_probe)`), so a flat bound there 408s a healthy
#: read, `_current_fit` comes back empty, the gate refuses the finish, and five refusals relabel a
#: fully fitted run `no_model_built`. Bounding once here keeps the queue term for reads too.
NON_FIT_CALL_TIMEOUT_S = 900.0

#: The judge's held-out score (:mod:`petab_mcp.validation_data`). A read on a fixed-θ set, but a
#: transfer set REFITS the model on it, so `_wrap_call_timeout` sizes it by :func:`_held_out_fits`.
HELD_OUT_TOOL = "evaluate_validation_data"


def _held_out_fits(example_dir: Path) -> int:
    """How many multistart fits :data:`HELD_OUT_TOOL` runs on the example's held-out set.

    Zero for a fixed-θ set, which is scored at the fitted point; one per refit unit for a transfer
    set (one when it names no units), since the server refits each unit on its own. Read with the
    server's own reader, so the two agree on what makes a set a transfer set: a ``parameters.tsv``
    beside the measurements. lucarelli's has three units, and under the read timeout its score
    never finished (gemot#57).

    Zero, too, for an absent or unreadable set: the server reads the same files at startup, so it
    has no refit to run either, and a timeout must never be the thing that fails a run.
    """
    measurements = validation_measurements_path(example_dir)
    if not measurements.is_file():
        return 0
    try:
        from petab_mcp.validation_data import MODE_TRANSFER, read_held_out

        held_out = read_held_out(measurements)
    except Exception as exc:  # noqa: BLE001 - sizing must never fail a run
        print(f"[runner] could not read the held-out set ({type(exc).__name__}: {exc})")
        return 0
    if held_out.mode != MODE_TRANSFER:
        return 0
    return len(set(held_out.units.values())) if held_out.units else 1


def _make_call(session: ClientSession):
    """Return an ``await call(tool, **args)`` bound to ``session`` (unwraps the result).

    Carries no timeout of its own: `_wrap_call_timeout` sizes every call, and a read gets
    `NON_FIT_CALL_TIMEOUT_S` plus whatever is queued ahead of it. An explicit budget here would
    suppress that queue term -- see the constant.

    An error result that reports an infrastructure failure (PEtab-MCP#95, or a raw full disk)
    raises :class:`~gemot.infrastructure.InfrastructureError` instead of being returned: unwrapped,
    it is a string where the judge expects a dict, and the run would end as a crash or a verdict
    on a broken server rather than as the infrastructure stop the sweep resubmits (gemot#39).
    """

    async def call(tool: str, /, **args):
        result = await session.call_tool(tool, args or {})
        value = _unwrap(result)
        if (
            getattr(result, "isError", False)
            and isinstance(value, str)
            and (hit := infrastructure.find(value)) is not None
        ):
            raise infrastructure.InfrastructureError(
                infrastructure.Evidence(hit.cause, f"{tool}: {hit.text}")
            )
        return value

    return call


#: Tools whose result carries an ``applied`` flag, i.e. an edit the server can refuse.
#: ``set_model``/``change_parameters`` are the pre-#130 names, kept so old traces and any
#: still-pinned server version are counted the same way.
_EDIT_TOOLS = frozenset({"make_changes", "set_model", "change_parameters"})

#: Rejection buckets, matched in order. Deliberately coarse: the point is to show WHICH KIND of
#: refusal dominates a run at a glance, so a regression (or a skill gap) is visible from
#: result.json without mining the trace DB. `antimony_syntax` and `invalid_problem` are the two
#: that mattered on the sweep this was built from — 37% and 59% of set_model refusals.
_REJECT_PATTERNS = (
    ("antimony_syntax", "failed to parse antimony"),
    ("invalid_problem", "resulting problem is invalid"),
    ("repeated_widening", "refusing to widen"),
    ("bad_bounds_usage", "sets only one bound"),
    ("unknown_key", "unknown key"),
    ("unknown_parameter", "unknown parameter id"),
    ("bad_scale", "invalid parameterscale"),
    ("bad_estimate", "cannot interpret estimate value"),
    ("missing_id", "must include an 'id'"),
    # NOTE there are deliberately no needles for the structured-op rejections. Those carry a
    # machine-readable `error_cause` (petab_mcp.operations.OperationError.cause), which
    # `note_result` prefers, so prose entries for them would be dead code — and a generic one
    # like "already exists" would additionally steal any FUTURE rejection containing the phrase,
    # which is the needle-shadows-the-bucket failure the labels were introduced to end.
    ("op_malformed", "nothing was applied"),  # residual, for anything unlabelled
)


def _result_payload(result: object) -> "dict | None":
    """The tool's JSON body out of an MCP result envelope, or ``None`` if it is not one."""
    try:
        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = (result.get("data") or {}).get("content")
        text = getattr(content[0], "text", None) if content else None
        if text is None and content:
            text = content[0].get("text")
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - a metric must never break a run
        return None


def _reject_cause(error: str) -> str:
    low = error.lower()
    for label, needle in _REJECT_PATTERNS:
        if needle in low:
            return label
    return "other"


class _ToolClock:
    """Client-observed wall-time during which at least one MCP tool call is in flight.

    This is time *as the runner sees it* — it includes the HTTP round-trip and any
    server-side queueing, not purely server execution — but for these demos it is
    dominated by the work inside the tool calls (AMICI compile + pyPESTO fits + SQL),
    so subtracting it from the wall clock is a good proxy for isolating LLM time.


    Tracks the UNION of busy intervals, not the sum of per-call durations: the LangGraph
    agent (deepagents' ToolNode) dispatches the tool calls in a single turn concurrently, so
    several ``session.call_tool`` coroutines overlap. Summing each one's duration double-counts the
    overlap and can exceed the wall clock (a fit + a SQL query running at once counts twice).
    A busy-interval measure instead answers "how much wall time was the server working," which
    is what we subtract from the wall clock to isolate LLM/inference time, and it can never
    exceed the wall clock.

    Also tallies REJECTED edits (see :meth:`note_result`), which is a different question from
    ``n_tool_errors``: a rejected edit is a SUCCESSFUL tool call returning ``applied: false``, so
    nothing upstream counts it. A Phoenix scan of one sweep found 518/2691 ``set_model`` calls
    (19%) and 111/1475 ``change_parameters`` calls (7.5%) rejected, dominated by causes the
    skills never mention — invisible in every result.json, and only discoverable by mining the
    36 GB trace DB after the fact. Carrying the rate per run makes it visible during a sweep.
    """

    __slots__ = (
        "seconds",
        "calls",
        "_active",
        "_since",
        "edits",
        "rejects",
        "fit_reverts",
        "reject_causes",
        "by_constraint",
        "first_commit_edit",
        "by_tool",
    )

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._active = 0  # in-flight call count
        self._since = 0.0  # monotonic time the busy window (0 -> 1) opened
        self.edits = 0
        self.rejects = 0
        self.fit_reverts = 0
        self.reject_causes: dict[str, int] = {}
        # Refusals per causal constraint, on the causal_constraints arm: which of the user's claims
        # the agent kept breaking. A refusal naming two counts for each.
        self.by_constraint: dict[str, int] = {}
        # Which editing call committed the run's first version (1-based), or None: how many calls
        # it took the agent to get a model past the server's checks.
        self.first_commit_edit: int | None = None
        # Per-tool CALL COUNTS. `calls` alone is a bare total, so "was this tool used at all?" was
        # answerable only from the Phoenix traces -- which need a shared collector running, and
        # otherwise land in `.slurm_scratch`, which is deleted when a task completes. A sweep run
        # without the collector is therefore permanently unable to answer the question, and one was:
        # the whole point of job 51614722 was whether the agent called `get_data_regimes`, and the
        # answer is unrecoverable. The name is already in hand at the single chokepoint every call
        # routes through; keeping it costs one dict and makes tool usage part of result.json, which
        # survives.
        self.by_tool: dict[str, int] = {}

    def enter(self, tool_name: "str | None" = None) -> None:
        if self._active == 0:
            self._since = time.monotonic()
        self._active += 1
        self.calls += 1
        if tool_name:
            self.by_tool[tool_name] = self.by_tool.get(tool_name, 0) + 1

    def leave(self) -> None:
        self._active -= 1
        if self._active == 0:
            self.seconds += time.monotonic() - self._since

    def note_result(self, tool_name: str, result: object) -> None:
        """Tally an EDITING tool call and, if the edit was refused, bucket why.

        Best-effort by construction: the result shape is the MCP envelope, and a change to it must
        degrade this metric, never break a run. Unparseable results simply are not counted.
        """
        if tool_name not in _EDIT_TOOLS:
            return
        payload = _result_payload(result)
        if payload is None:
            return
        self.edits += 1
        if self.first_commit_edit is None and payload.get("committed") is True:
            self.first_commit_edit = self.edits
        # `applied is False` ALONE is the edit verdict. The payload flattens the FIT's keys in
        # too, so an `error` can ride on a change that applied and was even snapshotted — testing
        # it counted those as refusals and polluted `by_cause` with fit failures, in the one
        # metric whose job is telling the two apart.
        if payload.get("applied", payload.get("committed")) is not False:
            return
        # Since #130 one call edits AND fits, so `applied: false` covers two different events. A
        # pure edit refusal is `{applied, error[, issues]}` straight from the tool's error wrapper;
        # once the fit stage has run, `bundle_result` always sets `committed`. Counting the fit
        # reverts as refusals made the rate an edit-or-fit failure rate and broke comparability
        # with the pre-#130 baseline — so they are tallied separately instead of discarded.
        if "committed" in payload:
            self.fit_reverts += 1
            return
        self.rejects += 1
        # The server's own label when it set one (petab_mcp.operations.OperationError.cause);
        # the prose match is the fallback for rejections raised deeper in the pipeline, which
        # carry no label.
        cause = payload.get("error_cause") or _reject_cause(str(payload.get("error") or ""))
        self.reject_causes[cause] = self.reject_causes.get(cause, 0) + 1
        if cause == CONSTRAINT_VIOLATED:
            for cid in broken_constraints(str(payload.get("error") or "")):
                self.by_constraint[cid] = self.by_constraint.get(cid, 0) + 1

    def reject_summary(self) -> dict:
        """``{attempted, rejected, reject_rate, by_cause}`` for ``result.json``."""
        return {
            "attempted": self.edits,
            "rejected": self.rejects,
            "fit_reverted": self.fit_reverts,
            "reject_rate": round(self.rejects / self.edits, 3) if self.edits else None,
            "by_cause": dict(sorted(self.reject_causes.items(), key=lambda kv: -kv[1])),
        }


def _wrap_call_timeout(
    session: ClientSession, default_n_starts: int, *, held_out_fits: int = 0
) -> None:
    """Give every ``session.call_tool`` a read timeout sized to THAT CALL's own ``n_starts``.

    This is where a fit timeout is actually decided. The client's ``sse_read_timeout`` is a
    per-SESSION httpx read timeout, so it cannot know any call's ``n_starts`` — and ``n_starts``
    is per call: ``make_changes`` (the only fitting tool) takes one that REPLACES the run
    default, the agent picks it at runtime, and ``build_true_model`` floors it higher still (
    bruno 400, fujita 300) on a scripted path that has no driver loop at all. So the session
    timeout is left deliberately non-binding (``runner.main``) and the exact cut is made here.

    Note the layering, which is easy to get backwards: ``read_timeout_seconds`` takes precedence
    per call, but it is an ``anyio.fail_after`` INSIDE the session — it can only end a call
    EARLIER than the httpx ceiling, never later. Raising a per-call budget above
    ``sse_read_timeout`` would achieve nothing; the ceiling has to be high and this has to be
    the tight one.

    Shadowing the instance method is how the runner already reaches every call site (see
    :func:`_wrap_tool_clock`): the runner's own calls AND the deepagents panel's route through
    the bound session, so no caller can bypass this — including ``build_true_model``, without
    ``Scenario`` having to surface its floor to the transport setup.

    A call carrying no ``n_starts`` (every read-only query) falls back to the run default, which
    is already generous for a SQL read. An explicit caller-supplied ``read_timeout_seconds`` is
    never overridden. The number itself comes from ``fit_budget.fit_time_budget_for``, shared
    with the stall watchdog so the two cannot disagree about how long a fit may take.

    ``held_out_fits`` is :func:`_held_out_fits` of the run's held-out set: the judge's
    :data:`HELD_OUT_TOOL` call fits that many times at the server's default ``n_starts``, and is
    sized as such a fit is. At 0, a fixed-θ set or none, it is a read like any other.
    """
    original = session.call_tool
    # n_starts of the fit calls currently in flight. `make_changes` holds `session.lock` for the
    # whole change-and-fit critical section (petab_mcp/server.py), so batched fit calls SERIALIZE
    # server-side: the second one waits out the first's entire fit on the lock before its own
    # starts. Its wait is therefore (everything ahead of it) + (its own fit) -- sizing it at its
    # own count alone would 408 a perfectly healthy call at roughly half the time it needs.
    in_flight: list[int] = []

    async def with_timeout(*args, **kwargs):
        # args[2] is `read_timeout_seconds` in ClientSession.call_tool(name, arguments,
        # read_timeout_seconds, progress_callback, *, meta). Check the positional slot too:
        # injecting the keyword on top of a positional one raises "got multiple values for
        # argument", which would make this wrapper the thing that fails the call.
        override = "read_timeout_seconds" in kwargs or len(args) > 2
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        asked = None
        # Parsed for EVERY call, not just the ones being sized: `in_flight` below has to see a
        # fit whoever set its timeout, or a concurrent read computes queued=0 against a server
        # that is busy.
        # Anything unexpected leaves `asked` None and takes the default budget rather than
        # raising -- a timeout must never be the thing that fails a call.
        with contextlib.suppress(Exception):
            # langchain-mcp-adapters passes (name, arguments) POSITIONALLY with only
            # progress_callback by keyword (langchain_mcp_adapters/tools.py), and so does
            # `_make_call`; the keyword form is accepted for any caller that uses it.
            tool_args = kwargs.get("arguments")
            if tool_args is None and len(args) > 1:
                tool_args = args[1]
            asked = (tool_args or {}).get("n_starts")
        # Could this call fit? The UNION of two signals, because each covers the other's
        # blind spot. `n_starts` present (whatever the tool) keeps the wrapper open to a
        # future fitting tool the day it is added rather than the day someone remembers to
        # list it -- see test_the_transport_wrapper_is_deliberately_tool_agnostic. A known
        # edit tool WITHOUT the argument is still a fit: `make_changes` takes the server's
        # default when the agent omits it, and a malformed payload leaves it unreadable, so
        # keying on presence alone would hand a real multistart a 900s budget. So would the
        # held-out score of a transfer set, which refits once per unit and carries no count.
        fits = held_out_fits if name == HELD_OUT_TOOL else 0
        can_fit = asked is not None or name in _EDIT_TOOLS or fits > 0
        if not override:
            # Queued behind whatever is already fitting (see `in_flight`). asyncio is
            # cooperative and there is no await between reading and appending, so the list is
            # consistent w.r.t. other calls arming at the same time.
            queued = sum(in_flight)
            if not can_fit:
                # A tool that cannot fit must not inherit a fit budget -- that is what let a
                # wedged read cost a whole multistart's wall clock, ten times over, on bachmann
                # 52859568_0. It still has to outwait anything ALREADY fitting, because the
                # server answers nothing while a sync `make_changes` holds its event loop, so the
                # queue term stays. (Dropping that term is what made two earlier per-handle
                # attempts unsafe: the finish gate reads from inside a super-step.)
                budget = NON_FIT_CALL_TIMEOUT_S + (
                    fit_time_budget_for(queued, default_n_starts) if queued else 0.0
                )
            else:
                budget = fit_time_budget_for(
                    None if asked is None else _plus(asked, queued),
                    default_n_starts,
                    fits=max(1, fits),
                )
            kwargs["read_timeout_seconds"] = timedelta(seconds=budget)
        tracked = None
        if can_fit:
            # Gated on `can_fit`, NOT on `asked is not None`. The server's own docstring tells the
            # model to omit `n_starts` unless it wants more than the default, so most real fits
            # arrive without one -- and while they were sized correctly for themselves, they were
            # invisible here, so a read dispatched alongside (langgraph fires a message's tool
            # calls together, and the finish gate reads from inside a super-step) computed
            # queued=0 and took a flat 900s against a server whose event loop was held for
            # thousands of seconds. That is the exact failure the queue term exists to stop.
            tracked = default_n_starts
            with contextlib.suppress(Exception):
                if asked is not None:
                    tracked = max(1, int(asked))
            tracked *= max(1, fits)
            in_flight.append(tracked)
        try:
            # Pass the caller's args through untouched otherwise: the adapter sends
            # `progress_callback=` on every single call and relies on it arriving.
            return await original(*args, **kwargs)
        finally:
            if tracked is not None:
                with contextlib.suppress(ValueError):
                    in_flight.remove(tracked)

    try:
        session.call_tool = with_timeout  # type: ignore[method-assign]
    except Exception as exc:
        # Best-effort like the clock, but say so. Without the shadow every call falls back to the
        # ClientSession-wide read_timeout_seconds `runner.main` sets, which is sized to the run
        # DEFAULT -- safe, but blind to a call asking for more. Print it so a sweep post-mortem
        # can tell that apart from a genuinely slow fit.
        print(
            f"[runner] could not install per-call fit timeouts ({type(exc).__name__}: {exc}) — "
            "tool calls fall back to the session-wide default-sized timeout"
        )


def _plus(asked, queued: int):
    """``asked + queued`` when ``asked`` is a usable count, else ``asked`` untouched.

    Keeps the coercion rules in ONE place (``fit_time_budget_for``): a value this cannot add to
    is passed through unchanged so that function still decides what to make of it.
    """
    if isinstance(asked, bool):
        return asked
    try:
        return max(1, int(asked)) + queued
    except (TypeError, ValueError, OverflowError):
        return asked


def _wrap_tool_clock(session: ClientSession) -> _ToolClock:
    """Time every ``session.call_tool`` into a returned clock (best-effort).

    Both the runner's own calls and the deepagents panel's calls route through
    ``session.call_tool`` (langchain-mcp-adapters invokes the bound session), so shadowing
    the instance method captures ALL tool time — which for these demos is dominated by the
    AMICI compile + every pyPESTO fit + the SQL queries. Subtracting it from the wall clock
    isolates the LLM/inference time. If the method can't be shadowed the clock simply stays
    at zero (timing is reported as unavailable) rather than breaking the run.

    asyncio is single-threaded and cooperative, so the ``enter``/``leave`` counter updates
    (no ``await`` between reading and mutating ``_active``) are atomic w.r.t. each other; the
    overlap is only ever inside the awaited ``original(...)``.
    """
    clock = _ToolClock()
    original = session.call_tool

    async def timed(*args, **kwargs):
        _name = kwargs.get("name") or (args[0] if args else None)
        clock.enter(_name if isinstance(_name, str) else None)
        try:
            result = await original(*args, **kwargs)
        finally:
            clock.leave()
        # Tally edit rejections here rather than at a call site: EVERY tool call routes through
        # this shadow (the runner's own and the agent panel's), so nothing can bypass the metric.
        with contextlib.suppress(Exception):
            # `_name` from above, not a second copy of the same expression: two derived metrics
            # (`by_tool` and `reject_causes`) must never disagree about what a call was called.
            if isinstance(_name, str):
                clock.note_result(_name, result)
        return result

    with contextlib.suppress(Exception):
        session.call_tool = timed  # type: ignore[method-assign]
    return clock
