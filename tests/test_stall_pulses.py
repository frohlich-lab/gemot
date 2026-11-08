"""An analyst round whose model calls keep ending is not a stall (gemot#136).

An analyst round is one super-step of the lead's graph, so its calls never moved the stall
watchdog. A round that ran past the window was cancelled, and the resume ran its unfinished `task`
calls again under the same tool call: byte-identical `task` calls in one turn, the cancelled one
with no output, with `dropped_dup_tool_calls` at 0 because the model had asked once.
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
    timer = loop.call_later(seconds, lambda: done.done() or done.set_result(None))
    try:
        await done
    finally:
        timer.cancel()


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


# -- the same on a real panel graph: what a cancelled round leaves behind ----------------------
def _panel(rounds: dict):
    """A deepagents panel with a checkpointer, as the driver builds it. The lead sends a
    mathematician `task` and a signal_engineer `task` in one turn, then finishes. The
    mathematician's FIRST round makes `_MATH_CALLS` model calls, each `_PANEL_CALL_S` long, so it
    runs past the window; any later round reports at once. ``rounds`` counts each analyst's
    rounds: one that starts over counts again, one picked up from its checkpoint does not."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.checkpoint.memory import MemorySaver

    from gemot.agent.guards import _tool_dispatch_guard, new_guard_stats

    stats = new_guard_stats()
    lead_turns = iter(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "fit numerics", "subagent_type": who},
                        "id": f"t-{who}",
                        "type": "tool_call",
                    }
                    for who in ("mathematician", "signal_engineer")
                ],
            ),
            AIMessage("done"),
        ]
    )

    async def reply(messages):
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        turns = sum(isinstance(m, AIMessage) for m in messages)
        if "PERSONA-LEAD" in system:
            return next(lead_turns)
        if "PERSONA" not in system:
            return AIMessage("summary so far")  # deepagents' summarizer, or anything unscripted
        who = "mathematician" if "PERSONA-MATH" in system else "signal_engineer"
        if turns == 0:
            rounds[who] = rounds.get(who, 0) + 1
        if who == "mathematician" and rounds[who] == 1:
            await _real_sleep(_PANEL_CALL_S)
            if turns < _MATH_CALLS - 1:
                call = {"name": "look", "args": {}, "id": f"look-{turns}", "type": "tool_call"}
                return AIMessage("", tool_calls=[call])
        return AIMessage(f"{who} report")

    class _Scripted(BaseChatModel):
        profile: dict | None = {"max_input_tokens": 1_000_000}

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise NotImplementedError("the driver runs the graph async")

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=await reply(messages))])

    look = StructuredTool.from_function(lambda: "seen", name="look", description="Look.")
    graph = create_deep_agent(
        model=_Scripted(),
        tools=[],
        system_prompt="PERSONA-LEAD",
        subagents=[
            {
                "name": "mathematician",
                "description": "numerics",
                "system_prompt": "PERSONA-MATH",
                "tools": [look],
            },
            {
                "name": "signal_engineer",
                "description": "signals",
                "system_prompt": "PERSONA-SIG",
                "tools": [],
            },
        ],
        middleware=[_tool_dispatch_guard(stats)],
        checkpointer=MemorySaver(),
    )
    return graph, stats


#: The panel tests' window, and the mathematician's first round: `_MATH_CALLS` model calls of
#: `_PANEL_CALL_S` each, twice the window. Wider than the tests above: a real graph's own steps
#: take longer than a fake one's.
_PANEL_WINDOW_S = 2.0
_PANEL_CALL_S = 0.2
_MATH_CALLS = 20


def _task_runs():
    """A callback recording each `task` run: its tool call, its analyst, and whether it returned
    an output. A run cancelled mid-flight returns none: langchain-core 1.7.3 reports nothing for
    it, and 1.7.6 reports a `CancelledError`. Either way the trace has no output for it."""
    from langchain_core.callbacks import AsyncCallbackHandler

    class _Runs(AsyncCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.started: dict = {}
            self.answered: set = set()

        async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
            if (serialized or {}).get("name") == "task":
                who = (kwargs.get("inputs") or {}).get("subagent_type")
                self.started[run_id] = (kwargs.get("tool_call_id"), who)

        async def on_tool_end(self, output, *, run_id, **kwargs):
            self.answered.add(run_id)

        def of(self, who):
            return [(tc, rid in self.answered) for rid, (tc, w) in self.started.items() if w == who]

    return _Runs()


async def _drive(graph, runs, pulses) -> int:
    """The driver's loop around ``astream``, cut to the stall watchdog and its resume. Returns how
    many times the run resumed from the checkpoint."""
    from langchain_core.messages import HumanMessage

    from gemot.agent import driver as D

    config = {
        "recursion_limit": 100,
        "configurable": {"thread_id": "run"},
        "callbacks": [runs] + ([pulses] if pulses is not None else []),
    }
    stream_input = {"messages": [HumanMessage("build a model")]}
    resumes = 0
    while True:
        agen = graph.astream(stream_input, config=config, stream_mode="values").__aiter__()
        try:
            while True:
                await D._next_step(agen, _PANEL_WINDOW_S, pulses=pulses)
        except StopAsyncIteration:
            return resumes
        except D._StepStalled:
            resumes += 1
            assert resumes <= 5, "the resumed round never finished"
            stream_input = None  # resume pending work from the checkpoint, as the driver does
        finally:
            await agen.aclose()


@requires_harness
@pytest.mark.parametrize("pulsed", [False, True])
def test_a_cancelled_round_runs_its_task_again_under_the_same_tool_call(pulsed):
    """Unpulsed, as before this change: the watchdog cancels the mathematician's round, and each
    resume runs that `task` again under the same tool call. Every cancelled copy has no output. The
    analyst picks its round up from its checkpoint rather than starting over, the
    signal_engineer's finished round is kept, and the dedup counts nothing, because the model
    asked once. Pulsed, the round's calls keep the watchdog off and the `task` runs once."""
    from gemot.agent import driver as D

    rounds: dict = {}
    graph, stats = _panel(rounds)
    runs = _task_runs()
    resumes = asyncio.run(_drive(graph, runs, D._llm_pulses() if pulsed else None))

    math = runs.of("mathematician")
    if pulsed:
        assert resumes == 0
        assert math == [("t-mathematician", True)]
    else:
        assert resumes >= 1
        assert math == [("t-mathematician", False)] * resumes + [("t-mathematician", True)]
    assert rounds == {"mathematician": 1, "signal_engineer": 1}
    assert runs.of("signal_engineer") == [("t-signal_engineer", True)]
    assert stats["dropped_dup_tool_calls"] == 0
