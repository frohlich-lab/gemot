"""Each fit result the lead reads closes with what the fit took and how much of the run is left.

gemot#137: on the 2026-11-08 sweep no fit result said either, and no agent asked
``get_run_timing``. Leads called a 4.3-h fit "the cheapest arbiter" and a 1.3-h refit "the cheap
decisive test", and on the problems whose fits take hours 14 of 15 runs sent no analyst beside a
fit. The driver-loop half, the line on the lead's own `make_changes`, is in
tests/test_run_loop_no_model.py.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from gemot import prompts as P
from gemot.agent import fit_cost as F

_FIT = {"applied": True, "committed": True, "version": 4, "bic": 812.3}


class _Tool:
    """Enough of a langchain tool for the wrapper: a name and a coroutine returning ``payload``
    the way the tool bridge does, as text blocks plus a ``structured_content`` artifact."""

    def __init__(self, payload, name="make_changes", artifact=True):
        self.name = name

        async def _run(**arguments):
            text = payload if isinstance(payload, str) else json.dumps(payload)
            sc = {"structured_content": payload} if artifact and isinstance(payload, dict) else None
            return ([{"type": "text", "text": text}], sc)

        self.coroutine = _run


def _call(tool, **arguments):
    return asyncio.run(tool.coroutine(**arguments))


@pytest.mark.parametrize(
    "seconds, span",
    [(12, "under a minute"), (90, "2 min"), (3540, "59 min"), (4.3 * 3600, "4.3 h")],
)
def test_a_span_reads_in_minutes_under_an_hour_and_hours_above(seconds, span):
    assert F._span(seconds) == span


@pytest.mark.parametrize("artifact", [True, False])
def test_a_fit_result_closes_with_its_cost_on_the_budget_clock(artifact):
    """The cost is what the budget clock moved over the call (the driver's clock, which leaves
    out the time a fit queued for the shared pool). The result is read from its artifact and,
    failing that, from its text."""
    clock = iter([1000.0, 1000.0 + 3600.0])
    tool = _Tool(_FIT, artifact=artifact)
    F._wrap_fit_cost(
        [tool], frozenset({"make_changes"}), spent=lambda: next(clock), left=lambda: 7200.0
    )

    content, _ = _call(tool, justification="j")

    assert content[0]["text"] == json.dumps(_FIT)  # the fit's own result, untouched
    assert content[-1] == {
        "type": "text",
        "text": "FIT COST: this fit took 1.0 h; 2.0 h of the run's time is left.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        # refused before the fit stage, in seconds: no `committed`
        {"applied": False, "error": "'k_pde' sets only one bound ... send both"},
        # the de-dup: the same model as a version already scored, so no fit ran
        {"applied": True, "committed": False, "no_change": True, "duplicate_of_version": 3},
        # the breaker's note in place of a fit that outran its call timeout: no result at all
        "FIT TIMED OUT: `make_changes` did not return in time",
    ],
)
def test_a_result_whose_fit_did_not_run_is_returned_as_it_came(payload):
    tool = _Tool(payload)
    expected = _call(tool, justification="j")
    F._wrap_fit_cost([tool], frozenset({"make_changes"}), spent=lambda: 0.0, left=lambda: 7200.0)

    assert _call(tool, justification="j") == expected


def test_a_reverted_fit_still_costs_what_it_took():
    """A fit that ran and could not be scored is reverted (``committed: false``); the run paid for
    it all the same."""
    tool = _Tool({"applied": False, "committed": False, "error": "every start failed"})
    F._wrap_fit_cost([tool], frozenset({"make_changes"}), spent=lambda: 0.0, left=lambda: 7200.0)

    content, _ = _call(tool, justification="j")

    assert content[-1]["text"].startswith("FIT COST: this fit took under a minute; 2.0 h")


def test_a_run_without_a_budget_hears_only_the_cost():
    assert F.fit_cost_note(5400.0, math.inf) == "FIT COST: this fit took 1.5 h."


def test_the_budget_left_never_reads_negative():
    """The run stops at its next step boundary once the budget is spent; a fit that ran past it
    reads none left, not a negative span."""
    assert F.fit_cost_note(60.0, -30.0).endswith("under a minute of the run's time is left.")


def test_only_the_fit_tools_are_wrapped():
    tool = _Tool(_FIT, name="get_version")
    expected = _call(tool, version=4)
    F._wrap_fit_cost([tool], frozenset({"make_changes"}), spent=lambda: 0.0, left=lambda: 7200.0)

    assert _call(tool, version=4) == expected


def test_the_lead_is_told_what_the_line_is_for_on_every_arm():
    """The line is read beside the rule that says what to do with it: price a refit by it, and
    send the analysts beside a long fit rather than in a turn of their own."""
    for lead in (P.DEFAULT_SYSTEM_PROMPT, P.INDRA_SYSTEM_PROMPT, P.BIOLOGY_BLIND_SYSTEM_PROMPT):
        assert P._FIT_COST_RULE in lead
