"""The progress_signals arm (``PETAB_PROGRESS_SIGNALS``): off, the agent is not told how well it is
doing, and nothing pushes it to go on or stop."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gemot.ablations import ABLATIONS
from gemot.runner.server import _SERVER_BOOTSTRAP

ROOT = Path(__file__).resolve().parents[1]

#: What the agent no longer reads when the arm is off: the server's withheld fields, and the
#: prompts' standing, verdicts and nudges.
_WITHHELD = (
    "is_new_best",
    "best_version",
    "chi2_gof_pvalue",
    "reduced_chi2",
    "get_progress",
    "bottomed out",
    "did not help costs",
    "did NOT help costs",
    "worse attempt",
    "time is left",
    "KEEP GOING",
    "KEEP ITERATING",
    "REFUSES",
    "the gate refuses",
    "the finish gate refuses",
    "fit is adequate",
    "accept the current structure and stop",
)

#: Texts the arm's off side never sends (the finish gate's refusals and review, the bounces and
#: the turn notices), and the null agent's own prompt, which the arm leaves alone.
_NOT_SENT_OFF = ("petab-finish-gate", "petab-nudges", "petab-guards", "petab-lead-indra-no-model")


def _fresh(script: str, **switches: str) -> str:
    env = {**os.environ, **{a.env: str(int(a.default_on)) for a in ABLATIONS}, **switches}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_the_arm_is_registered_on_and_moves_the_prompt():
    arm = next(a for a in ABLATIONS if a.name == "progress_signals")
    assert arm.env == "PETAB_PROGRESS_SIGNALS"
    assert arm.default_on is True and arm.changes_prompt is True


def _bootstrap(monkeypatch, argv: list[str]) -> dict:
    """Run the bootstrap with the server and uvicorn stubbed; what ``create_server`` was given."""
    seen: dict = {}
    server = types.ModuleType("petab_mcp.server")
    server.create_server = lambda *a, **kw: (
        seen.update(kw) or types.SimpleNamespace(run=lambda **_: None)
    )
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.Config = object
    monkeypatch.setitem(sys.modules, "petab_mcp.server", server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(sys, "argv", ["-c", *argv])
    exec(_SERVER_BOOTSTRAP, {})
    return seen


def test_the_server_gets_the_switch_as_the_last_argument(monkeypatch):
    argv = ["p.yaml", "build", "off", "127.0.0.1", "0", "5", "", "0", "", ""]
    assert _bootstrap(monkeypatch, [*argv, "1"])["progress_signals"] is True
    assert _bootstrap(monkeypatch, [*argv, "0"])["progress_signals"] is False
    with pytest.raises(ValueError, match="not enough values to unpack"):
        _bootstrap(monkeypatch, argv)


_SURFACE = r"""
import json
from gemot import prompts
from gemot.agent import circuit_breaker, fit_cost, guards
print(json.dumps({
    # The inlined petab-model-building skill warns against reduced_chi2 rather than pointing at it.
    "prompts": {
        k: v.replace(prompts._PETAB_SKILL_INLINE, "")
        for k, v in prompts.all_prompts().items()
        if "barebones" not in k
    },
    "groups": prompts.prompt_group_versions(),
    "fit_cost": fit_cost.fit_cost_note(60.0, 3600.0),
    "notice": guards._turn_notice(19, 20),
    "escalation": circuit_breaker._REJECTION_ESCALATION,
}))
"""


def test_off_no_text_the_agent_reads_points_at_a_withheld_signal():
    on = json.loads(_fresh(_SURFACE))
    off = json.loads(_fresh(_SURFACE, PETAB_PROGRESS_SIGNALS="0"))
    sent = {k: v for k, v in off["prompts"].items() if k not in _NOT_SENT_OFF}
    found = {(k, w) for k, v in sent.items() for w in _WITHHELD if w in v}
    assert not found
    assert all(any(w in v for v in on["prompts"].values()) for w in ("bottomed out", "KEEP GOING"))
    assert on["groups"]["lead"] != off["groups"]["lead"]
    assert off["prompts"]["petab-finish-description"].startswith("Ends the run, in one call.")
    assert "left" in on["fit_cost"] and off["fit_cost"] == "FIT COST: this fit took 1 min."
    assert on["notice"] and off["notice"] is None
    assert "accept the current structure" in on["escalation"]
    assert "accept the current structure" not in off["escalation"]


class _FakeResult:
    def __init__(self, payload):
        self.structured_content = payload


class _Session:
    """No stored version to restore, and HEAD's fit summary."""

    def __init__(self, fit_summary):
        self.fit_summary = fit_summary
        self.calls: list[str] = []

    async def call_tool(self, name, args):
        self.calls.append(name)
        if name == "list_versions":
            return _FakeResult({"result": []})
        if name == "get_fit_summary":
            return _FakeResult(self.fit_summary)
        raise AssertionError(f"unexpected tool {name}")


@pytest.fixture
def off(monkeypatch):
    pytest.importorskip("langchain_core", reason="harness-only dependency")
    from gemot import prompts

    monkeypatch.setattr(prompts, "PROGRESS_SIGNALS_ENABLED", False)


def _finish(fit_summary):
    from gemot import agent

    state: dict = {}
    session = _Session(fit_summary)
    tool = agent._finish_tool(session, state, adequacy="chi2")
    reply = asyncio.run(tool.ainvoke({"rationale": "done"}))
    return reply, state, session


def test_off_one_finish_ends_the_run_with_no_refusal_and_no_review(off):
    """A fit the gate would refuse on its sigma and on chi2, ended in one call."""
    reply, state, session = _finish(
        {
            "neg_log_likelihood": 10.0,
            "bic": 25.0,
            "criterion_value": 25.0,
            "version": 3,
            "noise_sigma_inflated": True,
            "chi2_gof_pvalue": 1e-9,
        }
    )
    assert (
        reply == "Recorded: finish — final bic=25, neg_log_likelihood=10. Report the bic, "
        "the neg_log_likelihood, and exactly what you changed, then stop."
    )
    assert state["confirmed"] is True and not state["refusals"]
    assert "get_fit_diagnosis" not in session.calls


def test_off_a_finish_without_a_fit_ends_the_run_as_no_model_built(off):
    from gemot import prompts

    reply, state, _ = _finish({"neg_log_likelihood": None, "bic": None, "version": None})
    assert reply == prompts.FINISH_ENDED_NO_FIT
    assert state["gave_up"] is True and state["confirmed"] is False


def test_off_the_cli_gateway_hides_get_progress(monkeypatch):
    pytest.importorskip("mcp")
    from mcp.types import Tool

    from gemot.agent import cli_mcp

    class _Upstream:
        async def list_tools(self, params=None):
            names = ("make_changes", "get_progress", "evaluate_validation_data")
            tools = [Tool(name=n, input_schema={"type": "object"}) for n in names]
            return types.SimpleNamespace(tools=tools, next_cursor=None)

    def listed():
        tools = asyncio.run(cli_mcp._listed_tools(_Upstream(), biology_blinded=False))
        return sorted(t.name for t in tools)

    assert listed() == ["get_progress", "make_changes"]
    monkeypatch.setenv("PETAB_PROGRESS_SIGNALS", "0")
    assert listed() == ["make_changes"]
