"""`build_true_model` is the shared scripted_fix body — and the only non-panel caller of the
editor's `parameters` argument.

The examples that pass `changes` (armistead, bruno, fujita) all fetch their PEtab problems
externally, so no test or smoke run exercises this path; a wrong kwarg here surfaces only in a
real benchmark run.
"""

from __future__ import annotations

import asyncio

from petab_harness.scenario import build_true_model


def _recorder():
    calls = []

    async def call(name, **kwargs):
        calls.append((name, kwargs))
        return {"applied": True, "committed": True, "version": 1, "fit": {}}

    return calls, call


def test_the_whole_build_is_one_call_with_the_tool_s_argument_names():
    entries = [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}]
    calls, call = _recorder()
    asyncio.run(build_true_model(
        call, antimony="model m end", justification="j", n_starts=5,
        changes=entries, change_justification="tighten",
    ))

    assert [n for n, _ in calls] == ["make_changes"]
    kwargs = calls[0][1]
    assert kwargs["antimony"] == "model m end"
    assert kwargs["parameters"] == entries
    assert kwargs["n_starts"] == 5
    assert "changes" not in kwargs, "the tool takes `parameters`"


def test_the_parameters_argument_is_omitted_when_there_is_nothing_to_retune():
    """`None`, not `[]`: an empty list is a malformed `parameters` argument, not "no edit"."""
    calls, call = _recorder()
    asyncio.run(build_true_model(call, antimony="model m end", justification="j", n_starts=5))
    assert [n for n, _ in calls] == ["make_changes"]
    assert calls[0][1]["parameters"] is None
