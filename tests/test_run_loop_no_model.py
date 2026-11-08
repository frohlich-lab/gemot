"""Driver-loop test for the no-model silent-stop guard in run_deepagents.

test_finish_gate.py unit-tests the _model_committed helper; this drives run_deepagents itself
with a fake agent whose graph "completes" without ever authoring a model, asserting the loop
(1) injects the build nudge and resumes, decrementing the resume budget, and (2) relabels the
run "no_model_built" once the budget is exhausted — the behaviour the finish gate cannot reach
because a no-tool turn ends the graph without calling the `finish` tool.

The same fake-driven run also pins the artifact contract: everything the driver RETURNS is
written to result.json, checked against the real return of a real run rather than against
runner/drive.py's source text.

Heavy examples-harness deps (deepagents, langchain_openai, langchain.mcp, openai) are
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
        for m in ("deepagents", "langchain_openai", "langchain.mcp", "openai", "langgraph")
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
        return type("R", (), {"structured_content": payload})()


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
    ``MCPError(REQUEST_TIMEOUT)``, the shape pinned by tests/test_judge_call_timeout.py."""

    def __init__(self):
        self.inputs = []
        self.captured = {}
        self.confirm_after = None

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)

        async def _gen():
            from mcp import MCPError
            from mcp.types import REQUEST_TIMEOUT

            raise MCPError(code=REQUEST_TIMEOUT, message="Timed out waiting for response")
            yield  # pragma: no cover - unreachable, present to make this an async generator

        return _gen()


def _run(
    mod,
    monkeypatch,
    probe,
    confirm_after=None,
    agent=None,
    server_poll=None,
    driver_options=None,
    coroutines=None,
):
    from langchain_core.tools import StructuredTool

    fake_agent = agent if agent is not None else _FakeAgent()
    fake_agent.confirm_after = confirm_after

    # Capture the finish_state the driver hands the finish tool, so the fake agent can flip it
    # to confirmed (a fake astream can't itself call `finish`).
    _real_finish_tool = mod._finish_tool

    def _capturing_finish_tool(session, finish_state=None, probe=None, **gate):
        fake_agent.captured["fs"] = finish_state
        return _real_finish_tool(session, finish_state, probe=probe, **gate)

    monkeypatch.setattr(mod.driver, "_finish_tool", _capturing_finish_tool)

    # Real (inert) tool objects named per ALLOWED_TOOLS so _select_tools' curation passes; the
    # faked create_deep_agent ignores them, so they are never actually invoked -- except the ones
    # given ``coroutines``, which a fake agent can call through the real tool wrapper.
    coroutines = coroutines or {}
    fake_tools = [
        StructuredTool.from_function(
            func=(lambda: ""), coroutine=coroutines.get(n), name=n, description=n
        )
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
    from gemot.agent import mcp_tools as lmt

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
    assert result["silent_bounces"] == mod._MAX_SILENT_BOUNCES
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


@requires_harness
@pytest.mark.parametrize("confirm_after, bounces", [(1, 0), (2, 1)])
def test_the_result_counts_silent_bounces(monkeypatch, confirm_after, bounces):
    """gemot#124: in sweep 63372750, fontaine 8 and 9 were each bounced right after the finish
    gate's third sigma refusal, and neither result showed it. The count lived in a local, and
    `idle_nudges` (no-tool turns before the first tool call) read 0. A run that finishes first
    time carries an explicit 0, so a missing key never stands for "not bounced"."""
    mod = _load_agent_module()
    result, fake_agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), confirm_after=confirm_after
    )

    assert len(fake_agent.inputs) == 1 + bounces
    assert result["silent_bounces"] == bounces
    assert result["idle_nudges"] == 0  # the counter this was mistaken for does not see it


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


# --- a fit that outruns its call timeout -----------------------------------------------------
# gemot#56: a 150-start fit hung on the server (PEtab-MCP#108), and each of three resumes re-sent
# it into the server still holding it, a full 7560s timeout apiece. The wrapper's half is in
# tests/test_fit_call_timeout.py; these drive the loop around it.


def _timing_out_fit(n_timeouts):
    """A `make_changes` whose first ``n_timeouts`` calls time out the way mcp reports it."""

    async def _make_changes(**kw):
        from mcp import MCPError
        from mcp.types import REQUEST_TIMEOUT

        _make_changes.calls += 1
        if _make_changes.calls <= n_timeouts:
            raise MCPError(code=REQUEST_TIMEOUT, message="Timed out. Waited 7560.0 seconds.")
        return ([{"type": "text", "text": '{"applied": true}'}], None)

    _make_changes.calls = 0
    return _make_changes


class _FitAgent(_TimeoutAgent):
    """Graph stub whose first astream calls the lead's `make_changes` ``fits`` times through the
    real tool wrapper, as the tool node would, records what came back, and finishes."""

    def __init__(self, fits):
        super().__init__()
        self.fits, self.results = fits, []

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)
        captured, results, fits = self.captured, self.results, self.fits

        async def _gen():
            make_changes = next(t for t in captured["tools"] if t.name == "make_changes")
            for _ in range(fits):
                results.append(await make_changes.coroutine(n_starts=150))
                yield {"messages": [None] * len(results)}
            captured["fs"]["confirmed"] = True

        return _gen()


@requires_harness
def test_a_fit_timeout_reaches_the_lead_instead_of_a_resume(monkeypatch):
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_FitAgent(fits=2),
        coroutines={"make_changes": _timing_out_fit(1)},
    )

    assert len(agent.inputs) == 1  # never resumed
    assert agent.results[0][0].startswith("FIT TIMED OUT: `make_changes` did not return in time")
    assert agent.results[1][0][0]["text"] == '{"applied": true}'  # the lead's next fit ran
    assert result["stop_reason"] == "completed"


@requires_harness
def test_a_second_fit_timeout_ends_the_run_as_a_wedged_fit(monkeypatch):
    """Rather than spending the resumes: each would re-send the fit for another full timeout."""
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_FitAgent(fits=2),
        coroutines={"make_changes": _timing_out_fit(2)},
        server_poll=lambda: None,
    )

    assert result["stop_reason"] == "infrastructure_error: fit_wedged"
    assert len(agent.inputs) == 1  # no resume
    assert result["infrastructure_evidence"].startswith("make_changes: MCPError: Timed out.")
    assert "behind make_changes: MCPError" in result["infrastructure_evidence"]
    assert result["make_changes_calls"]["by_outcome"] == {
        "tool_error": 1,
        "infrastructure_error": 1,
    }


@requires_harness
def test_the_leads_fit_result_closes_with_its_cost(monkeypatch):
    """gemot#137: the lead's own `make_changes`, through every wrapper the driver puts on it, ends
    with what the fit took and how much of the run's wall budget is left."""

    async def _make_changes(**kw):
        return ([{"type": "text", "text": '{"applied": true, "committed": true}'}], None)

    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_FitAgent(fits=1),
        coroutines={"make_changes": _make_changes},
        driver_options={"max_wall_seconds": 36_000},
    )

    content, _ = agent.results[0]
    assert content[0]["text"] == '{"applied": true, "committed": true}'
    assert content[-1]["text"] == (
        "FIT COST: this fit took under a minute; 10.0 h of the run's time is left."
    )
    assert result["stop_reason"] == "completed"


# --- a client-side MCPError (gemot#103) --------------------------------------------------------
# f2e2b86d ended 15.8 h into a 33 h budget on one `get_version` reply too large for the client to
# read. The wrapper's half, and the error's real shape, are in tests/test_client_mcp_error.py.


class _ReadAgent(_TimeoutAgent):
    """Graph stub whose first astream calls the lead's `get_version` twice through the real tool
    wrapper, as the tool node would, records what came back, and finishes."""

    def __init__(self):
        super().__init__()
        self.results = []

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)
        captured, results = self.captured, self.results

        async def _gen():
            get_version = next(t for t in captured["tools"] if t.name == "get_version")
            for _ in range(2):
                results.append(await get_version.coroutine(version=23, include=["simulation"]))
                yield {"messages": [None] * len(results)}
            captured["fs"]["confirmed"] = True

        return _gen()


@requires_harness
def test_a_reply_the_client_could_not_read_reaches_the_lead_and_the_run_goes_on(monkeypatch):
    from mcp import MCPError
    from mcp.types import CONNECTION_CLOSED

    too_large = "SSE stream failed: Server-sent event exceeded the 1048576 byte limit."

    async def _get_version(**kw):
        _get_version.calls += 1
        if _get_version.calls == 1:
            raise MCPError(code=CONNECTION_CLOSED, message=too_large)
        return ([{"type": "text", "text": '{"version": 23}'}], None)

    _get_version.calls = 0
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_ReadAgent(),
        coroutines={"get_version": _get_version},
    )

    assert agent.results[0] == (f"MCPError: {too_large}", None)
    assert agent.results[1][0][0]["text"] == '{"version": 23}'  # the lead's next call ran
    assert len(agent.inputs) == 1  # no resume
    assert result["stop_reason"] == "completed"


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


# --- a full disk -----------------------------------------------------------------------------
# gemot#39: on a full quota the INDRA sweep's runs were resumed and bounced like any other, and
# their leads re-authored models that were fine. Whichever witness sees the disk, nothing the
# loop normally does to keep a run alive may happen.


def _evidence():
    from gemot.infrastructure import Evidence

    return Evidence(
        "disk_full", "amici_build.log: /usr/bin/ld: final link failed: Disk quota exceeded"
    )


class _DiskFullAgent(_TimeoutAgent):
    """Graph stub whose every astream raises what the tool wrapper raises on a full disk."""

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)

        async def _gen():
            from gemot.infrastructure import Evidence, InfrastructureError

            raise InfrastructureError(
                Evidence("disk_full", "make_changes: infrastructure_error: disk_full: [Errno 122]")
            )
            yield  # pragma: no cover - unreachable, present to make this an async generator

        return _gen()


@requires_harness
def test_the_tool_wrappers_stop_ends_the_run_without_a_resume(monkeypatch):
    mod = _load_agent_module()
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_DiskFullAgent(), server_poll=None
    )

    assert result["stop_reason"] == "infrastructure_error: disk_full"
    assert result["infrastructure_evidence"].startswith("make_changes: infrastructure_error")
    assert len(agent.inputs) == 1  # where an unknown error would have ended it too, unlabelled


@requires_harness
@pytest.mark.parametrize(
    "agent_class",
    [
        _TimeoutAgent,  # a fit timeout against a live server: resumed 3 times otherwise
        _FakeAgent,  # a silent stop with a model: bounced 3 times otherwise
    ],
)
def test_the_logs_end_the_run_where_the_tools_saw_nothing(monkeypatch, agent_class):
    """The 241 rejections said only "ill-posed"; the build log named the disk."""
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=agent_class(),
        driver_options={"infrastructure_probe": _evidence},
    )

    assert result["stop_reason"] == "infrastructure_error: disk_full"
    assert result["infrastructure_evidence"] == _evidence().text
    assert result["infrastructure_tool_calls"] == 0
    assert len(agent.inputs) == 1


@requires_harness
def test_a_clean_run_reports_no_full_disk(monkeypatch):
    mod = _load_agent_module()
    result, _agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        driver_options={"infrastructure_probe": lambda: None},
    )
    assert result["stop_reason"] == "silent_stop"
    assert result["infrastructure_tool_calls"] == 0 and result["infrastructure_evidence"] is None


# --- a call timeout is not a stall (gemot#55) --------------------------------------------------
# On 2026-10-21 a ~14-minute OpenRouter outage on z-ai/glm-5.3-flash (four 120s chunk timeouts
# and a 429) ended two INDRA runs with 17-20h of their wall budget left: the watchdog's
# `except asyncio.TimeoutError` caught langchain's StreamChunkTimeoutError, a TimeoutError
# subclass, so each call timeout read as a stall, doubled the stall window and spent a resume.


class _ScriptedAgent(_FakeAgent):
    """Graph stub that plays one scripted outcome per astream call, then behaves as `_FakeAgent`:
    "chunk" raises langchain's StreamChunkTimeoutError, "timeout" a bare TimeoutError, "hang"
    emits nothing until the watchdog ends it, and "step" emits two states, so it is progress
    even as the first emission of a resume."""

    def __init__(self, *script):
        super().__init__()
        self.script = list(script)

    def astream(self, stream_input, config=None, stream_mode=None):
        self.inputs.append(stream_input)
        what = self.script.pop(0) if self.script else "step"
        n = len(self.inputs)

        async def _gen():
            import asyncio

            if what == "chunk":
                from langchain_openai import StreamChunkTimeoutError

                raise StreamChunkTimeoutError(120.0, model_name="z-ai/glm-5.3-flash")
            if what == "timeout":
                raise TimeoutError("a call inside the step timed out")
            if what == "credit":
                import httpx
                import openai

                request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
                text = "You requested up to 131072 tokens, but can only afford 91509."
                raise openai.APIError(text, request, body={"code": 402, "message": text})
            if what == "hang":
                await asyncio.Event().wait()
            yield {"messages": [{"role": "user", "content": "x"}] * (2 * n)}
            yield {"messages": [{"role": "user", "content": "x"}] * (2 * n + 1)}

        return _gen()


def _record_windows(mod, monkeypatch) -> list:
    """Record the window the driver arms for each step, but let a hung step expire at once."""
    real_next_step = mod.driver._next_step
    windows = []

    async def _recording(agen, timeout, **_):
        windows.append(timeout)
        return await real_next_step(agen, 0.01)

    monkeypatch.setattr(mod.driver, "_next_step", _recording)
    return windows


@requires_harness
@pytest.mark.parametrize("inner", ["chunk", "timeout"])
def test_a_timeout_inside_the_step_is_an_outage_not_a_stall(monkeypatch, inner):
    """The inner timeout is waited out as a provider outage: the next step runs at the SAME
    window, and only the real stall after it doubles it."""
    mod = _load_agent_module()
    windows = _record_windows(mod, monkeypatch)
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_ScriptedAgent(inner, "hang")
    )

    base = mod.driver._STEP_STALL_TIMEOUT_S
    assert windows[:3] == [base, base, 2 * base]
    assert result["provider_outage_retries"] == 1
    assert result["provider_outage_wait_s"] == mod.driver._OUTAGE_BACKOFF_S
    assert result["stop_reason"] == "silent_stop"  # ran on to its normal end


@requires_harness
def test_a_real_watchdog_expiry_is_still_a_stall(monkeypatch):
    mod = _load_agent_module()
    windows = _record_windows(mod, monkeypatch)
    resumes = mod.driver._RUN_RESUME_ATTEMPTS
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_ScriptedAgent(*["hang"] * (1 + resumes)),
    )

    base = mod.driver._STEP_STALL_TIMEOUT_S
    assert windows == [base * 2**i for i in range(1 + resumes)]
    assert result["stop_reason"].startswith(f"stalled: no step completed in {windows[-1]:.0f}s")
    assert result["provider_outage_retries"] == 0
    assert len(agent.inputs) == 1 + resumes


@requires_harness
def test_an_outage_burst_does_not_end_the_run(monkeypatch):
    """More outages than there are resumes, twice over, with a step between the bursts: each
    burst stays inside the outage budget, and the step between them resets it."""
    mod = _load_agent_module()
    budget = mod.driver._OUTAGE_RETRY_ATTEMPTS
    assert budget > mod.driver._RUN_RESUME_ATTEMPTS
    burst = ["chunk"] * budget
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_ScriptedAgent(*burst, "step", *burst)
    )

    assert result["stop_reason"] == "silent_stop"
    assert result["provider_outage_retries"] == 2 * budget
    assert result["provider_outage_wait_s"] == 2 * sum(
        mod.driver._outage_backoff(i) for i in range(1, budget + 1)
    )


@requires_harness
def test_an_outage_past_the_budget_ends_the_run(monkeypatch):
    mod = _load_agent_module()
    budget = mod.driver._OUTAGE_RETRY_ATTEMPTS
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_ScriptedAgent(*["chunk"] * (budget + 1)),
    )

    assert result["stop_reason"].startswith("stream: StreamChunkTimeoutError")
    assert result["provider_outage_retries"] == budget
    assert len(agent.inputs) == 1 + budget


@requires_harness
def test_a_402_ends_the_run_as_provider_credit_without_a_retry(monkeypatch):
    """gemot#76: a mid-stream 402 is the bare APIError the outage path waits out; the same request
    is refused until the account is topped up, so the run ends at once with a countable label."""
    mod = _load_agent_module()
    result, agent = _run(
        mod, monkeypatch, _Probe(model_authored=True), agent=_ScriptedAgent("credit")
    )

    assert result["stop_reason"] == mod.driver.PROVIDER_CREDIT
    assert result["provider_outage_retries"] == 0
    assert len(agent.inputs) == 1


@requires_harness
def test_the_wall_budget_bounds_an_outage_wait(monkeypatch):
    """With 50s to the last step a run waits 30s for the first retry, but not 60s for the second:
    it could not start a step after that wait, so it stops there with a budget label."""
    mod = _load_agent_module()
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_ScriptedAgent("chunk", "chunk"),
        driver_options={"max_wall_seconds": 100.0},  # headroom: half the budget, 50s
    )

    assert result["stop_reason"].startswith(
        "wall_time_budget_exceeded (during retry: stream: StreamChunkTimeoutError"
    )
    assert result["provider_outage_retries"] == 1
    assert len(agent.inputs) == 2


@requires_harness
def test_time_queued_for_the_fit_pool_is_not_the_runs(monkeypatch):
    """gemot#106: the run of the test above, but with 100s of it spent queued for a shared fit
    pool. That wait moves the end of the budget out, so the second outage wait still fits before
    the last step, and the run goes on to its normal end."""
    from gemot.runner.clock import _Heartbeats

    mod = _load_agent_module()
    heartbeats = _Heartbeats()
    heartbeats.queue_seconds = 100.0
    result, agent = _run(
        mod,
        monkeypatch,
        _Probe(model_authored=True),
        agent=_ScriptedAgent("chunk", "chunk"),
        driver_options={"max_wall_seconds": 100.0, "heartbeats": heartbeats},
    )

    assert result["stop_reason"] == "silent_stop"
    assert result["provider_outage_retries"] == 2
