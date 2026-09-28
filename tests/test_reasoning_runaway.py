"""The reasoning-runaway detector and its retry ladder.

Built against the REAL langchain ModelResponse class, not a stand-in: the first version of the
detector read ``.messages`` off a dataclass whose field is ``.result`` and returned False on
every call — an error a test with a hand-rolled fake would have reproduced instead of caught.
"""

import asyncio

import pytest

pytest.importorskip("langchain")

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from petab_harness.agent.guards import (
    _is_reasoning_runaway,
    _reasoning_runaway_guard,
    new_guard_stats,
)


def _resp(content="", finish="length", tool_calls=None) -> ModelResponse:
    msg = AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        response_metadata={"finish_reason": finish},
    )
    return ModelResponse(result=[msg], structured_response=None)


# ---------------------------------------------------------------- detection


def test_detects_the_measured_glm_signature():
    # finish_reason="length", content "", no tool call — the exact shape recorded on
    # z-ai/glm-5.3-flash (completion_tokens=65536, reasoning_tokens=65504).
    assert _is_reasoning_runaway(_resp(content="", finish="length"))


def test_whitespace_only_content_is_a_runaway():
    assert _is_reasoning_runaway(_resp(content="\n  \n", finish="length"))


def test_textless_content_blocks_are_a_runaway():
    # Reasoning models return content as a list of blocks; a truncated turn can carry a
    # reasoning block with no text.
    blocks = [{"type": "reasoning", "reasoning": "step 1: consider the"}]
    assert _is_reasoning_runaway(_resp(content=blocks, finish="length"))


def test_a_tool_call_is_never_a_runaway():
    calls = [{"name": "make_changes", "args": {}, "id": "1", "type": "tool_call"}]
    assert not _is_reasoning_runaway(_resp(content="", finish="length", tool_calls=calls))


def test_a_clean_stop_is_not_a_runaway():
    assert not _is_reasoning_runaway(_resp(content="done", finish="stop"))


def test_truncated_but_substantive_text_is_not_a_runaway():
    # length-terminated with real content: the turn was cut short but said something a
    # downstream consumer can act on — retrying it would discard that.
    assert not _is_reasoning_runaway(_resp(content="The model needs a Hill term", finish="length"))


def test_block_content_with_text_is_not_a_runaway():
    blocks = [{"type": "text", "text": "partial answer"}]
    assert not _is_reasoning_runaway(_resp(content=blocks, finish="length"))


# ---------------------------------------------------------------- retry ladder


class _Request:
    """The two things the guard touches on a ModelRequest: .model_settings and .override()."""

    def __init__(self, model_settings=None):
        self.model_settings = model_settings or {}

    def override(self, **kwargs):
        new = _Request(kwargs.get("model_settings", self.model_settings))
        return new


def test_ladder_retries_at_reduced_effort_and_recovers():
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)
    seen_efforts: list = []

    async def handler(request):
        seen_efforts.append((request.model_settings or {}).get("reasoning_effort"))
        # runaway on the original call, healthy on the first retry
        return _resp(content="") if len(seen_efforts) == 1 else _resp(content="ok", finish="stop")

    out = asyncio.run(guard.awrap_model_call(_Request(), handler))
    assert not _is_reasoning_runaway(out)
    assert seen_efforts == [None, "low"]  # original untouched; retry stepped down one rung
    assert stats["reasoning_runaways"] == 1
    assert stats["reasoning_runaway_recovered"] == 1


def test_ladder_is_bounded_when_every_effort_fails():
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return _resp(content="")  # runaway at every effort

    out = asyncio.run(guard.awrap_model_call(_Request(), handler))
    assert _is_reasoning_runaway(out)  # handed back honestly, not swallowed
    assert calls == 3  # original + the two ladder rungs, then stop
    # ONE event, however many rungs it took: `runaways` counts detections, `recovered` the
    # subset a retry rescued, so recovered/runaways is a real recovery rate in result.json.
    assert stats["reasoning_runaways"] == 1
    assert stats["reasoning_runaway_recovered"] == 0


def test_sync_hook_shares_the_ladder():
    # The sync wrap_model_call is unreachable under the astream-only driver but is part of the
    # middleware contract; it must behave identically to the async hook.
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)
    seen_efforts: list = []

    def handler(request):
        seen_efforts.append((request.model_settings or {}).get("reasoning_effort"))
        return _resp(content="") if len(seen_efforts) == 1 else _resp(content="ok", finish="stop")

    out = guard.wrap_model_call(_Request(), handler)
    assert not _is_reasoning_runaway(out)
    assert seen_efforts == [None, "low"]
    assert stats["reasoning_runaways"] == 1
    assert stats["reasoning_runaway_recovered"] == 1


def test_a_healthy_final_message_wins_over_an_earlier_truncated_one():
    # .result may carry more than one message. The verdict belongs to the TERMINAL model turn;
    # convicting on an earlier one made the ladder discard a good answer and re-call the model.
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    healthy = AIMessage(
        content="the model needs a Hill term", response_metadata={"finish_reason": "stop"}
    )
    assert not _is_reasoning_runaway(
        ModelResponse(result=[truncated, healthy], structured_response=None)
    )


def test_a_healthy_final_turn_without_finish_reason_still_wins():
    # Not every provider/middleware sets finish_reason. Deciding on "the last message that has
    # one" let an earlier truncated turn convict a healthy final answer — the terminal test is
    # the last MODEL turn, whatever metadata it carries.
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    healthy = AIMessage(content="needs a Hill term", response_metadata={})
    assert not _is_reasoning_runaway(
        ModelResponse(result=[truncated, healthy], structured_response=None)
    )


def test_a_trailing_tool_message_is_skipped():
    # The ToolMessage langchain appends when a tool produced the structured output: not a model
    # turn, so the verdict comes from the AIMessage before it. A real ToolMessage, not an
    # AIMessage stand-in — the skip is by message TYPE now, so a stand-in would test nothing.
    truncated = AIMessage(content="", response_metadata={"finish_reason": "length"})
    trailing = ToolMessage(content="{}", tool_call_id="1")
    assert _is_reasoning_runaway(
        ModelResponse(result=[truncated, trailing], structured_response=None)
    )


def test_a_failing_retry_falls_back_to_the_original_turn():
    # A provider that rejects reasoning_effort must not turn a survivable idle turn into a dead
    # run: the guard swallows the error and hands back the original truncated result.
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _resp(content="")
        raise ValueError("400: unsupported parameter reasoning_effort")

    out = asyncio.run(guard.awrap_model_call(_Request(), handler))
    assert _is_reasoning_runaway(out)  # the original, handed back
    assert stats["reasoning_runaway_retry_errors"] == 1  # visible, not silent
    assert stats["reasoning_runaway_recovered"] == 0


def test_healthy_call_passes_through_untouched():
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return _resp(content="fine", finish="stop")

    asyncio.run(guard.awrap_model_call(_Request(), handler))
    assert calls == 1
    assert stats["reasoning_runaways"] == 0


# ------------------------------------------------- panel coverage + replay detection


def test_every_subagent_carries_the_runaway_guard():
    # deepagents builds each subagent its OWN middleware stack, so the lead's list does not
    # reach the analysts — 11 of 32 measured runaways were on subagents.
    from petab_harness.agent.biocurator import _build_biocurator_subagent
    from petab_harness.agent.subagents import _build_subagents

    stats: dict = {}
    specs = _build_subagents({}, stats) + [_build_biocurator_subagent([], [], stats)]
    # By NAME, not identity: the factory defines a fresh local class per call, so each spec's
    # guard is a distinct type object from any other instance's.
    wanted = type(_reasoning_runaway_guard({})).__name__
    for spec in specs:
        kinds = [type(m).__name__ for m in spec["middleware"]]
        assert wanted in kinds, f"{spec['name']} has no runaway guard: {kinds}"


def test_subagents_build_without_stats():
    # The stats argument is optional so the builders stay usable (and tested) standalone.
    from petab_harness.agent.subagents import _build_subagents

    for spec in _build_subagents({}):
        assert spec["middleware"]  # the turn limit is still there


def test_state_message_count_distinguishes_unreadable_from_empty():
    # The watchdog's replay test: None (unreadable) must NOT be conflated with 0 (real, empty),
    # or one odd emission would leave the stall window permanently widened.
    from petab_harness.agent.driver import _state_message_count

    assert _state_message_count({"messages": [1, 2, 3]}) == 3
    assert _state_message_count({"messages": []}) == 0
    assert _state_message_count({}) is None
    assert _state_message_count("not a state") is None
    assert _state_message_count({"messages": "nope"}) is None
