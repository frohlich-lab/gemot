"""`build_true_model` is the shared scripted_fix body — and the only caller of
`change_parameters` outside the panel.

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
        return {"applied": True, "committed": True, "version": 1}

    return calls, call


def test_change_parameters_is_called_with_the_tool_s_argument_name():
    entries = [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}]
    calls, call = _recorder()
    asyncio.run(build_true_model(
        call, antimony="model m end", justification="j", n_starts=5,
        changes=entries, change_justification="tighten",
    ))

    name, kwargs = next(c for c in calls if c[0] == "change_parameters")
    assert kwargs["parameters"] == entries
    assert "changes" not in kwargs, "the tool takes `parameters`, the same name set_model takes"


def test_no_parameter_call_when_there_is_nothing_to_retune():
    calls, call = _recorder()
    asyncio.run(build_true_model(call, antimony="model m end", justification="j", n_starts=5))
    assert [n for n, _ in calls] == ["set_model", "fit"]
