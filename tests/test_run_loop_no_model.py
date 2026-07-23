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
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


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
    # Import as the real _harness.agent package module (NOT the isolation-by-path loader the
    # other tests use): run_deepagents does relative imports (from .fit_budget import ...) that
    # only resolve with the package context.
    if str(_EXAMPLES) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES))
    from _harness import agent
    return agent


class _Probe:
    """Fake un-timed progress_probe: model never authored, run never asked to early-stop."""

    def __init__(self, model_initialized=False):
        self.model_initialized = model_initialized

    async def __call__(self, tool, args):
        payload = ({"model_initialized": self.model_initialized} if tool == "describe_problem"
                   else {"should_stop": False})
        return type("R", (), {"structuredContent": payload})()


class _FakeAgent:
    """Graph stub whose astream yields ONE super-step then ends — a clean 'completed' with
    whatever the probe reports as the model state. Records each astream input so the test can
    see the resume nudge."""

    def __init__(self):
        self.inputs = []

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)

        async def _gen():
            yield {"messages": []}
        return _gen()


def _run(mod, monkeypatch, probe):
    from langchain_core.tools import StructuredTool
    fake_agent = _FakeAgent()

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
def test_no_model_run_nudges_resumes_then_relabels(monkeypatch):
    mod = _load_agent_module()
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_initialized=False))

    # Never built a model -> relabelled, not an ambiguous "completed".
    assert result["stop_reason"] == "no_model_built"
    # Initial drive + one resume per _RUN_RESUME_ATTEMPTS.
    assert len(fake_agent.inputs) == 1 + mod._RUN_RESUME_ATTEMPTS
    # The resumes injected the build nudge as a fresh user turn (the first input is the task).
    resume_inputs = fake_agent.inputs[1:]
    assert resume_inputs, "expected at least one resume"
    for inp in resume_inputs:
        msg = inp["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == mod._NO_MODEL_NUDGE


@requires_harness
def test_run_that_authors_a_model_completes_without_resume(monkeypatch):
    mod = _load_agent_module()
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_initialized=True))

    # A run that authored a model must NOT be relabelled empty and must NOT be resumed. (The
    # exact label is "completed"; a faked graph runs no real tools, so the orthogonal, pre-
    # existing zero-tools relabel may make it "no_tools_executed" — either way, not our label.)
    assert result["stop_reason"] != "no_model_built"
    assert len(fake_agent.inputs) == 1              # no resume
