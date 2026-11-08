"""An analyst round whose model calls keep ending is not a stall (gemot#136).

An analyst round is one super-step of the lead's graph, so its calls never moved the stall
watchdog. A round that ran past the window was cancelled, and the resume ran its unfinished `task`
calls again under the same tool call: byte-identical `task` calls in one turn, one with no output,
with `dropped_dup_tool_calls` at 0 because the model had asked once.
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import requires_langchain
from test_run_loop_no_model import _FakeAgent, _Probe, _load_agent_module, _run, requires_harness

#: The window the tests run at, and how often a model call ends inside a step. A step runs for
#: `_STEP_S`, past the window and past the one second the driver's first wait always gets.
_WINDOW_S = 0.5
_CALL_S = 0.05
_STEP_S = 2.0


async def _real_sleep(seconds: float) -> None:
    """A wait that takes real time even where a test has replaced ``asyncio.sleep``."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.call_later(seconds, done.set_result, None)
    await done


async def _model_call(callbacks) -> None:
    """One model call that ends, through langchain's own callback path."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    await FakeListChatModel(responses=["ok"]).ainvoke("hi", config={"callbacks": callbacks})


async def _step(seconds: float, callbacks=None):
    """A super-step that takes ``seconds``; with ``callbacks``, model calls keep ending in it."""
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        await _real_sleep(_CALL_S)
        if callbacks is not None:
            await _model_call(callbacks)
    yield "stepped"


@requires_langchain
def test_model_calls_that_end_push_the_watchdog_out():
    from gemot.agent import driver as D

    async def run():
        pulses = D._llm_pulses()
        await _model_call([pulses])  # the first call imports, off the clock
        step = _step(_STEP_S, callbacks=[pulses]).__aiter__()
        return await D._next_step(step, _WINDOW_S, pulses=pulses)

    assert asyncio.run(run()) == "stepped"


@requires_langchain
def test_with_no_call_ending_the_watchdog_still_fires():
    """A call hung mid-stream ends nothing, so it is still a stall."""
    from gemot.agent import driver as D

    async def run():
        pulses = D._llm_pulses()
        return await D._next_step(_step(_STEP_S).__aiter__(), _WINDOW_S, pulses=pulses)

    with pytest.raises(D._StepStalled):
        asyncio.run(run())


class _AnalystRound(_FakeAgent):
    """A first super-step that runs past the window, as an analyst round does; with ``calls``,
    the model calls inside it keep ending, through the callbacks the driver configured."""

    def __init__(self, calls: bool):
        super().__init__()
        self.calls = calls

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)
        callbacks = (config or {}).get("callbacks") if self.calls else None
        first = len(self.inputs) == 1

        async def _gen():
            if first:
                async for _ in _step(_STEP_S, callbacks=callbacks):
                    pass
            yield {"messages": []}

        return _gen()


def _short_window(mod, monkeypatch) -> None:
    real_next_step = mod.driver._next_step

    async def _short(agen, timeout, **kwargs):
        return await real_next_step(agen, _WINDOW_S, **kwargs)

    monkeypatch.setattr(mod.driver, "_next_step", _short)


@requires_harness
@pytest.mark.parametrize("calls", [True, False])
def test_a_round_still_making_calls_is_not_resumed(monkeypatch, calls):
    """The driver hands the watchdog the calls of the whole graph. The round without calls is the
    control: the same round, silent, is cancelled and resumed (a resume streams from ``None``)."""
    mod = _load_agent_module()
    _short_window(mod, monkeypatch)
    asyncio.run(_model_call([mod.driver._llm_pulses()]))  # the first call imports, off the clock
    agent = _AnalystRound(calls)
    _run(mod, monkeypatch, _Probe(model_authored=True), agent=agent)
    assert (None in agent.inputs) is not calls
