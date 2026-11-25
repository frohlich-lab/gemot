"""A provider refusal of one analyst's request ends that analyst's round, not the run (gemot#166).

The ``task`` calls of one lead turn are one super-step. A refusal escaping one of them (luna's
content filter, a rejected request) failed the super-step, cancelled the other analysts' rounds,
and ended the run, because the driver does not resume what the provider will refuse again.
"""

from __future__ import annotations

import asyncio

import pytest

from test_run_loop_no_model import requires_harness

pytest.importorskip("openai", reason="harness-only dependency")

import httpx  # noqa: E402
import openai  # noqa: E402

from gemot import prompts  # noqa: E402


def _status_error(cls, status: int, text: str, body=None):
    response = httpx.Response(status, request=httpx.Request("POST", "http://x"))
    return cls(text, response=response, body=body)


_FILTERED = "400 This content was flagged for possible biological risk."


@pytest.mark.parametrize(
    "exc",
    [
        _status_error(openai.BadRequestError, 400, _FILTERED),
        _status_error(openai.BadRequestError, 400, "context length exceeded"),
        _status_error(openai.UnprocessableEntityError, 422, "invalid tool schema"),
        # OpenRouter's mid-stream error event: the bare APIError, its code in the body.
        openai.APIError(
            "upstream refused", request=httpx.Request("POST", "http://x"), body={"code": 400}
        ),
        # An HTTP-200 error body the client raises as a ValueError.
        ValueError("{'error': {'message': 'bad request', 'code': 400}}"),
    ],
)
def test_a_refusal_of_one_request_is_the_analysts(exc):
    from gemot.agent.provider_refusals import analyst_refusal

    assert analyst_refusal(exc) is not None


@pytest.mark.parametrize(
    "exc",
    [
        # Credit, a key or a model that fails every call: the run's, and it must end as before.
        _status_error(openai.APIStatusError, 402, "can only afford 91509 tokens"),
        _status_error(openai.AuthenticationError, 401, "invalid key"),
        _status_error(openai.NotFoundError, 404, "no endpoints found"),
        # Transient: the driver waits it out and resumes each analyst from its checkpoint.
        _status_error(openai.RateLimitError, 429, "slow down"),
        _status_error(openai.InternalServerError, 502, "bad gateway"),
        openai.APIError("upstream died", request=httpx.Request("POST", "http://x"), body=None),
        TimeoutError("stream went silent"),
        # Not a provider error at all: a bug stays loud.
        KeyError("messages"),
    ],
)
def test_anything_else_is_left_to_the_run_loop(exc):
    from gemot.agent.provider_refusals import analyst_refusal

    assert analyst_refusal(exc) is None


@requires_harness
def test_every_analyst_and_the_biocurator_carry_it_innermost():
    """Innermost, so the runaway guard's own retries are covered too."""
    from gemot.agent.biocurator import _build_biocurator_subagent
    from gemot.agent.guards import new_guard_stats
    from gemot.agent.subagents import _build_subagents

    stats = new_guard_stats()
    specs = _build_subagents({}, stats) + [_build_biocurator_subagent([], stats=stats)]
    assert {type(spec["middleware"][-1]).__name__ for spec in specs} == {"_ProviderRefusalReport"}
    assert stats["subagent_provider_refusals"] == 0  # seeded, so result.json carries the zero


def _panel(stats: dict):
    """A deepagents panel, as the driver builds it: the lead sends a mathematician `task` and a
    signal_engineer `task` in one turn, then ends with what it was handed. The signal_engineer's
    provider refuses its first call; the mathematician reports after a beat, so its round is still
    running when the refusal comes."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from gemot.agent.provider_refusals import _provider_refusal_report

    async def reply(messages):
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        if "PERSONA-LEAD" in system:
            reports = [m for m in messages if isinstance(m, ToolMessage)]
            if reports:
                return AIMessage("done")
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "look", "subagent_type": who},
                        "id": f"t-{who}",
                        "type": "tool_call",
                    }
                    for who in ("mathematician", "signal_engineer")
                ],
            )
        if "PERSONA-SIG" in system:
            raise _status_error(openai.BadRequestError, 400, _FILTERED)
        if "PERSONA-MATH" in system:
            await asyncio.sleep(0.3)
            return AIMessage("mathematician report")
        return AIMessage("summary so far")  # deepagents' summarizer, or anything unscripted

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

    def analyst(name, persona):
        return {
            "name": name,
            "description": name,
            "system_prompt": persona,
            "tools": [],
            "middleware": [_provider_refusal_report(stats, name)],
        }

    return create_deep_agent(
        model=_Scripted(),
        tools=[],
        system_prompt="PERSONA-LEAD",
        subagents=[
            analyst("mathematician", "PERSONA-MATH"),
            analyst("signal_engineer", "PERSONA-SIG"),
        ],
    )


@requires_harness
def test_one_analysts_refusal_leaves_the_others_report_and_the_run():
    from langchain_core.messages import HumanMessage, ToolMessage

    stats: dict = {}
    state = asyncio.run(
        _panel(stats).ainvoke(
            {"messages": [HumanMessage("build a model")]}, {"recursion_limit": 50}
        )
    )
    reports = {m.tool_call_id: m.content for m in state["messages"] if isinstance(m, ToolMessage)}
    assert reports["t-mathematician"] == "mathematician report"
    refused = reports["t-signal_engineer"]
    assert refused.startswith(prompts.SUBAGENT_PROVIDER_REFUSAL_TMPL.split("{")[0])
    assert "flagged for possible biological risk" in refused
    assert state["messages"][-1].content == "done"
    assert stats["subagent_provider_refusals"] == 1
