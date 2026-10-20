"""Driver-loop test for the no-model silent-stop guard in run_deepagents.

test_finish_gate.py unit-tests the _model_committed helper; this drives run_deepagents itself
with a fake agent whose graph "completes" without ever authoring a model, asserting the loop
(1) injects the build nudge and resumes, decrementing the resume budget, and (2) relabels the
run "no_model_built" once the budget is exhausted — the behaviour the finish gate cannot reach
because a no-tool turn ends the graph without calling the `finish` tool.

The same fake-driven run also pins the artifact contract: everything the driver RETURNS is
written to result.json, checked against the real return of a real run rather than against
runner/drive.py's source text.

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
    not all(
        _have(m)
        for m in ("deepagents", "langchain_openai", "langchain_mcp_adapters", "openai", "langgraph")
    ),
    reason="examples-harness deps (deepagents/langchain/openai) not installed",
)


def _load_agent_module():
    from gemot import agent

    return agent


class _Probe:
    """Fake un-timed progress_probe: model never authored, run never asked to early-stop."""

    def __init__(self, model_authored=False):
        self.model_authored = model_authored

    async def __call__(self, tool, args):
        payload = (
            {"model_authored": self.model_authored}
            if tool == "describe_problem"
            else {"should_stop": False}
        )
        return type("R", (), {"structuredContent": payload})()


class _FakeAgent:
    """Graph stub whose astream yields ONE super-step then ends — a clean 'completed' with
    whatever the probe reports as the model state. Records each astream input so the test can
    see the resume nudge. If ``confirm_after`` is set, it flips the captured finish_state to
    confirmed on that astream call, simulating the model finishing through the gate."""

    def __init__(self):
        self.inputs = []
        self.captured = {}  # {"fs": finish_state} populated by the capturing _finish_tool
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


class _TimeoutAgent:
    """Graph stub whose every astream raises the per-call fit timeout mcp really raises — a bare
    ``McpError(408)``, the shape pinned by tests/test_judge_call_timeout.py."""

    def __init__(self):
        self.inputs = []
        self.captured = {}
        self.confirm_after = None

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)

        async def _gen():
            from mcp.shared.exceptions import McpError
            from mcp.types import ErrorData

            raise McpError(ErrorData(code=408, message="Timed out waiting for response"))
            yield  # pragma: no cover - unreachable, present to make this an async generator

        return _gen()


def _run(
    mod, monkeypatch, probe, confirm_after=None, agent=None, server_poll=None, driver_options=None
):
    from langchain_core.tools import StructuredTool

    fake_agent = agent if agent is not None else _FakeAgent()
    fake_agent.confirm_after = confirm_after

    # Capture the finish_state the driver hands the finish tool, so the fake agent can flip it
    # to confirmed (a fake astream can't itself call `finish`).
    _real_finish_tool = mod._finish_tool

    def _capturing_finish_tool(session, finish_state=None, probe=None):
        fake_agent.captured["fs"] = finish_state
        return _real_finish_tool(session, finish_state, probe=probe)

    monkeypatch.setattr(mod.driver, "_finish_tool", _capturing_finish_tool)

    # Real (inert) tool objects named per ALLOWED_TOOLS so _select_tools' curation passes; the
    # faked create_deep_agent ignores them, so they are never actually invoked.
    fake_tools = [
        StructuredTool.from_function(func=(lambda: ""), name=n, description=n)
        for n in sorted(mod.ALLOWED_TOOLS)
    ]

    async def _fake_load_mcp_tools(session):
        return fake_tools

    def _fake_create_deep_agent(
        *,
        model=None,
        tools=None,
        system_prompt=None,
        subagents=None,
        middleware=None,
        checkpointer=None,
        name=None,
        **kw,
    ):
        fake_agent.captured.update(
            system_prompt=system_prompt, subagents=subagents, tools=tools, name=name
        )
        return fake_agent

    import deepagents
    import langchain_mcp_adapters.tools as lmt

    monkeypatch.setattr(lmt, "load_mcp_tools", _fake_load_mcp_tools)
    monkeypatch.setattr(deepagents, "create_deep_agent", _fake_create_deep_agent)

    # The resume backoff is real time (5 + 10 + 15 = 30s for a run that exhausts the budget), and
    # nothing here is testing that we can sleep. Without this, the three liveness tests below add
    # ~60s to every suite run.
    async def _no_backoff(_seconds):
        return None

    monkeypatch.setattr(mod.driver.asyncio, "sleep", _no_backoff)

    import asyncio

    options = dict(
        model="fake/model",
        base_url="http://x",
        api_key="sk-test",
        system_prompt="sp",
        task="build a model",
        recursion_limit=50,
        progress_probe=probe,
        server_poll=server_poll,
    )
    options.update(driver_options or {})
    result = asyncio.run(
        mod.run_deepagents(
            object(),  # session — only passed to the (mocked) load_mcp_tools and decision tool
            **options,
        )
    )
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
    for inp in fake_agent.inputs[1:]:  # every bounce = the build nudge
        msg = inp["messages"][0]
        assert msg["role"] == "user" and msg["content"] == mod._NO_MODEL_NUDGE


@requires_harness
def test_model_silent_stop_bounces_into_the_gate(monkeypatch):
    """A silent stop WITH a model (no finish through the gate) is bounced with the FINISH
    nudge — the gate cannot be sidestepped — and relabelled 'silent_stop', not 'completed'."""
    mod = _load_agent_module()
    result, fake_agent = _run(mod, monkeypatch, _Probe(model_authored=True))

    assert result["stop_reason"] == "silent_stop"  # not an accepted finish
    assert len(fake_agent.inputs) == 1 + mod._MAX_SILENT_BOUNCES
    for inp in fake_agent.inputs[1:]:  # every bounce = the finish nudge
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
    assert len(fake_agent.inputs) == 2  # initial + 1 bounce, then confirmed


# --- liveness at the CALL SITE ---------------------------------------------------------------
# `_server_is_alive` is unit-tested in tests/test_judge_call_timeout.py; these drive the branch
# in run_deepagents that consumes it, which is where the three-state handling actually matters.


@requires_harness
def test_a_dead_server_ends_the_run_without_spending_a_resume(monkeypatch):
    """bachmann 52859568_0: the OOM killer took the server down and the loop resumed into it
    three more times, each re-sending the whole history and re-issuing the pending fit for
    another full 4200s timeout. The check must come BEFORE the resume budget is spent."""
    mod = _load_agent_module()
    agent = _TimeoutAgent()
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=agent, server_poll=lambda: -9
    )

    assert result["stop_reason"].startswith("server_unreachable:")
    assert "fit call timed out" in result["stop_reason"]  # keeps WHY it stopped talking
    assert len(agent.inputs) == 1  # no resume attempted


@requires_harness
def test_a_live_server_still_gets_every_resume_attempt(monkeypatch):
    """The other side: a fit timeout against a server that is still running is exactly the case
    the checkpoint resume exists for, and must not be cut short."""
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_TimeoutAgent(),
        server_poll=lambda: None,
    )

    assert not result["stop_reason"].startswith("server_unreachable")
    assert len(agent.inputs) == 1 + mod.driver._RUN_RESUME_ATTEMPTS


@requires_harness
def test_no_liveness_signal_does_not_end_the_run(monkeypatch):
    """`None` is 'cannot tell', not 'dead'. A refactor to `if not await _server_is_alive(...)`
    would end every run that wires no signal — this is the test that would catch it."""
    mod = _load_agent_module()
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_TimeoutAgent(), server_poll=None
    )

    assert not result["stop_reason"].startswith("server_unreachable")
    assert len(agent.inputs) == 1 + mod.driver._RUN_RESUME_ATTEMPTS


# --- the context-window stop ------------------------------------------------------------------


def _past_the_window(mod, monkeypatch):
    """Make the run's accountant report a largest prompt over 0.90 x the default window."""
    monkeypatch.delenv("PETAB_MCP_MAX_INPUT_TOKENS", raising=False)
    real_accountant = mod.driver._token_accountant

    def _accountant():
        accountant = real_accountant()
        accountant.max_prompt = 1_000_402  # > 0.90 x the 1,048,576-token default window
        return accountant

    monkeypatch.setattr(mod.driver, "_token_accountant", _accountant)


@requires_harness
def test_the_context_window_stop_is_a_budget_stop(monkeypatch):
    """gemot#7: the 0.90 x window stop used to be recorded as "early_stop: context_window_
    approached: largest prompt 1,000,402 tokens ...", which `stop_class` filed as the server's
    plateau stop and whose counts matched the "402" out-of-credits marker. It is a budget stop
    with a bare label, and its numbers stay readable from their own fields."""
    from gemot import analysis, experiments

    mod = _load_agent_module()
    _past_the_window(mod, monkeypatch)
    result, agent = _run(mod, monkeypatch, _Probe(model_authored=True))

    assert result["stop_reason"] == "context_budget_exceeded"
    assert len(agent.inputs) == 1  # stopped, not bounced or resumed
    assert analysis.stop_class(result) == "budget"
    assert not experiments.is_noop_infra_abort({**result, "n_ops": 0, "n_fits": 0})
    assert result["token_usage"]["max_prompt_tokens"] == 1_000_402
    assert result["max_input_tokens"] == 1_048_576


@requires_harness
def test_a_stall_past_the_window_stops_instead_of_resuming(monkeypatch):
    """A step that crossed the line and then timed out never reaches the boundary check. Without
    the retry-path guard it resumed every attempt, re-sending ~1M tokens each time, and ended as
    "fit call timed out: ..." -- an infrastructure stop, not the budget stop it was."""
    mod = _load_agent_module()
    _past_the_window(mod, monkeypatch)
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_TimeoutAgent(), server_poll=None
    )

    assert result["stop_reason"] == "context_budget_exceeded"
    assert len(agent.inputs) == 1  # no resume attempted


# --- the artifact contract -------------------------------------------------------------------


@requires_harness
def test_result_json_carries_every_key_the_driver_returns(monkeypatch, tmp_path):
    """A measurement the driver reports and result.json drops is invisible to every sweep.

    `agent_out` here is a REAL run_deepagents return and the writer is the real one, so this
    fails on a key the artifact loses — which is what a per-field copy in the runner did to
    `openrouter_endpoints` for its whole life (docs/RATIONALE.md). Both writers spread `meta`
    (runner/judge.py and _write_degraded_result), so either proves the propagation.
    """
    import json
    from types import SimpleNamespace

    from gemot.runner.failure import _write_degraded_result
    from gemot.runner.provenance import _agent_meta

    mod = _load_agent_module()
    agent_out, graph = _run(mod, monkeypatch, _Probe(model_authored=True))
    assert graph.captured["name"] == "lead"

    meta = _agent_meta({"driver": "deepagents"}, agent_out, agent_out["stop_reason"])
    _write_degraded_result(tmp_path, meta, SimpleNamespace(name="toy"), "test")
    written = json.loads((tmp_path / "result.json").read_text())

    missing = sorted(k for k in agent_out if k not in written)
    assert not missing, f"the driver reported these and result.json dropped them: {missing}"
    assert written["token_usage"]["final_lead_prompt_tokens"] is None
    assert written["token_usage"]["max_lead_prompt_tokens"] is None
