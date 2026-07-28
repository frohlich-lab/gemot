"""Driver-loop test for the no-model silent-stop guard in run_deepagents.

test_finish_gate.py unit-tests the _model_committed helper; this drives run_deepagents itself
with a fake agent whose graph "completes" without ever authoring a model, asserting the loop
(1) injects the build nudge and resumes, decrementing the resume budget, and (2) relabels the
run "no_model_built" once the budget is exhausted — the behaviour the finish gate cannot reach
because a no-tool turn ends the graph without calling make_decision.

Heavy examples-harness deps (deepagents, langchain_openai, langchain_mcp_adapters, openai) are
monkeypatched/needed; skip cleanly where absent.
"""

from __future__ import annotations

import importlib.util

import pytest


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


requires_harness = pytest.mark.skipif(
    not all(_have(m) for m in ("deepagents", "langchain_openai",
                               "langchain_mcp_adapters", "openai", "langgraph")),
    reason="examples-harness deps (deepagents/langchain/openai) not installed",
)


def _load_agent_module():
    from petab_harness import agent

    return agent


class _Probe:
    """Fake un-timed progress_probe: model never authored, run never asked to early-stop."""

    def __init__(self, model_authored=False):
        self.model_authored = model_authored

    async def __call__(self, tool, args):
        payload = ({"model_authored": self.model_authored} if tool == "describe_problem"
                   else {"should_stop": False})
        return type("R", (), {"structuredContent": payload})()


class _FakeAgent:
    """Graph stub whose astream yields ONE super-step then ends — a clean 'completed' with
    whatever the probe reports as the model state. Records each astream input so the test can
    see the resume nudge. If ``confirm_after`` is set, it flips the captured finish_state to
    confirmed on that astream call, simulating the model finishing through make_decision."""

    def __init__(self):
        self.inputs = []
        self.captured = {}          # {"fs": finish_state} populated by the capturing _decision_tool
        self.confirm_after = None

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)
        n = len(self.inputs)
        captured, confirm_after = self.captured, self.confirm_after

        async def _gen():
            if confirm_after and n >= confirm_after and captured.get("fs") is not None:
                captured["fs"]["confirmed"] = True
            yield {"messages": []}
        return _gen()


def _run(mod, monkeypatch, probe, confirm_after=None):
    from langchain_core.tools import StructuredTool
    fake_agent = _FakeAgent()
    fake_agent.confirm_after = confirm_after

    # Capture the finish_state the driver hands the decision tool, so the fake agent can flip it
    # to confirmed (a fake astream can't itself call make_decision).
    _real_decision_tool = mod._decision_tool

    def _capturing_decision_tool(session, finish_state=None, probe=None):
        fake_agent.captured["fs"] = finish_state
        return _real_decision_tool(session, finish_state, probe=probe)

    monkeypatch.setattr(mod.driver, "_decision_tool", _capturing_decision_tool)

    # Real (inert) tool objects named per ALLOWED_TOOLS so _select_tools' curation passes; the
    # faked create_deep_agent ignores them, so they are never actually invoked.
    fake_tools = [StructuredTool.from_function(func=(lambda: ""), name=n, description=n)
                  for n in sorted(mod.ALLOWED_TOOLS)]

    async def _fake_load_mcp_tools(session):
        return fake_tools

    def _fake_create_deep_agent(*, model=None, tools=None, system_prompt=None,
                                subagents=None, middleware=None, checkpointer=None, **kw):
        return fake_agent

    import deepagents
    import langchain_mcp_adapters.tools as lmt
    monkeypatch.setattr(lmt, "load_mcp_tools", _fake_load_mcp_tools)
    monkeypatch.setattr(deepagents, "create_deep_agent", _fake_create_deep_agent)

    import asyncio
    result = asyncio.run(mod.run_deepagents(
        object(),  # session — only passed to the (mocked) load_mcp_tools and decision tool
        model="fake/model", base_url="http://x", api_key="sk-test",
        system_prompt="sp", task="build a model", recursion_limit=50,
        progress_probe=probe,
    ))
    return result, fake_agent


@requires_harness
def test_no_model_silent_stop_bounces_with_build_nudge(monkeypatch):
    """A silent stop that authored no model is bounced back with the BUILD nudge, up to
    _MAX_SILENT_BOUNCES, then relabelled 'no_model_built'. The fake never confirms a finish, so
    it is treated as a silent stop every time."""
    mod = _load_agent_module()
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_authored=False))

    assert result["stop_reason"] == "no_model_built"
    assert len(fake_agent.inputs) == 1 + mod._MAX_SILENT_BOUNCES  # initial drive + N bounces
    for inp in fake_agent.inputs[1:]:                             # every bounce = the build nudge
        msg = inp["messages"][0]
        assert msg["role"] == "user" and msg["content"] == mod._NO_MODEL_NUDGE


@requires_harness
def test_model_silent_stop_bounces_into_the_gate(monkeypatch):
    """A silent stop WITH a model (no finish through make_decision) is bounced with the FINISH
    nudge — the gate cannot be sidestepped — and relabelled 'silent_stop', not 'completed'."""
    mod = _load_agent_module()
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_authored=True))

    assert result["stop_reason"] == "silent_stop"                # not an accepted finish
    assert len(fake_agent.inputs) == 1 + mod._MAX_SILENT_BOUNCES
    for inp in fake_agent.inputs[1:]:                            # every bounce = the finish nudge
        msg = inp["messages"][0]
        assert msg["role"] == "user" and msg["content"] == mod._SILENT_FINISH_NUDGE


@requires_harness
def test_confirmed_finish_is_not_bounced(monkeypatch):
    """A run that finishes through the gate (finish_state confirmed) must NOT be bounced: it ends
    as 'completed' the moment the confirm lands, not re-driven."""
    mod = _load_agent_module()
    # First turn stops silently -> one bounce; on the 2nd turn the model "confirms" -> done.
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_authored=True), confirm_after=2)
    # A confirmed finish is respected: NOT relabelled as a bypassed gate, and NOT re-bounced.
    # (The exact label may be the orthogonal pre-existing "no_tools_executed" — the faked graph
    # runs no real tools — but never our silent-stop labels.)
    assert result["stop_reason"] not in ("silent_stop", "no_model_built")
    assert len(fake_agent.inputs) == 2                          # initial + 1 bounce, then confirmed
