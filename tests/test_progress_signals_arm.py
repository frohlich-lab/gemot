"""The progress_signals arm (``PETAB_PROGRESS_SIGNALS``): off, the agent gets none of the numbers
that tell it how well it is doing, and the server's streak or plateau does not stop the run."""

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

#: What the agent no longer reads when the arm is off: the server's withheld fields and the
#: numbers the harness quotes.
_WITHHELD = (
    "is_new_best",
    "best_version",
    "chi2_gof_pvalue",
    "reduced_chi2",
    "get_progress",
    "time is left",
    "refuses a finish on a scored fit that fails chi2",
)

#: The finish gate's and the guards' own texts, checked by what they send rather than by name.
_NOT_SENT_OFF = ("petab-finish-gate", "petab-guards")


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
    argv = ["p.yaml", "build", "off", "127.0.0.1", "0", "5", "", "0", "", "", "1"]
    assert _bootstrap(monkeypatch, [*argv, "1"])["progress_signals"] is True
    assert _bootstrap(monkeypatch, [*argv, "0"])["progress_signals"] is False
    with pytest.raises(ValueError, match="not enough values to unpack"):
        _bootstrap(monkeypatch, argv)


_SURFACE = r"""
import json
from gemot import prompts
from gemot.agent import fit_cost, guards
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
}))
"""


def test_off_no_text_the_agent_reads_names_a_withheld_number():
    on = json.loads(_fresh(_SURFACE))
    off = json.loads(_fresh(_SURFACE, PETAB_PROGRESS_SIGNALS="0"))
    sent = {k: v for k, v in off["prompts"].items() if k not in _NOT_SENT_OFF}
    assert not {(k, w) for k, v in sent.items() for w in _WITHHELD if w in v}
    lead = on["prompts"]["petab-lead-system"]
    assert "chi2_gof_pvalue" in lead and "time is left" in lead
    assert "fails chi2" in on["prompts"]["petab-finish-description"]
    assert on["groups"]["lead"] != off["groups"]["lead"]
    assert "left" in on["fit_cost"] and off["fit_cost"] == "FIT COST: this fit took 1 min."
    # Turns left are the harness's budget, not progress: withheld, delegations ran into the hard
    # cap on this arm and nowhere else (gemot#159).
    assert on["notice"] == off["notice"] and "FINAL TURN" in off["notice"]


#: The finish gate, armed once with HEAD at v5 (the gate restores v4, the best) and once with HEAD
#: already at v4.
_FINISH = r"""
import asyncio
import json
from gemot import agent, prompts

class R:
    def __init__(self, payload):
        self.structured_content = payload

BIC = {4: 205.0, 5: 260.0}

class Session:
    def __init__(self, head):
        self.head = head

    async def call_tool(self, name, args):
        if name == "list_versions":
            return R({"result": [{"version": v, "criterion_value": b, "bic": b} for v, b in BIC.items()]})
        if name == "describe_problem":
            return R({"current_version": self.head})
        if name == "get_version":
            return R({"version": args["version"]})
        if name == "restore_version":
            self.head = args["version"]
            return R({"version": self.head})
        if name == "get_fit_summary":
            b = BIC[self.head]
            return R({"neg_log_likelihood": b / 2 - 10, "bic": b, "criterion_value": b,
                      "version": self.head, "noise_sigma_report": {}})
        if name == "get_fit_diagnosis":
            ident = {"available": True, "fim": {"rank_deficient": False}, "flat_directions": []}
            return R({"per_observable": [], "identifiability": ident})
        raise AssertionError(name)

def armed(head):
    return asyncio.run(agent._finish_tool(Session(head)).ainvoke({"rationale": "done"}))

print(json.dumps({"moved": armed(5), "stayed": armed(4), "description": prompts.FINISH_DESCRIPTION,
                  "note": prompts.FINISH_RESTORED_BEST_TMPL.format(best=4, head=5)}))
"""


def test_off_a_finish_says_neither_which_version_is_best_nor_whether_head_was():
    """gemot#156: the restore note named the best version on this arm, in 20 of 30 runs, and the
    ARM step quoted its bic, which names it to a lead keeping a ledger of its fits' scores."""
    pytest.importorskip("langchain_core", reason="harness-only dependency")
    on = json.loads(_fresh(_FINISH))
    off = json.loads(_fresh(_FINISH, PETAB_PROGRESS_SIGNALS="0"))
    assert on["moved"].startswith(on["note"]) and "v4" in on["note"] and "bic=205" in on["moved"]
    assert "restored" not in on["stayed"] and "bic=205" in on["stayed"]
    assert "current bic" in on["description"]
    assert off["moved"] == off["stayed"]
    assert off["moved"].startswith(off["note"]) and "FINISH ARMED" in off["moved"]
    assert not [w for w in ("v4", "v5", "205", "best-scoring") if w in off["moved"]]
    assert "current bic" not in off["description"]


class _FakeResult:
    def __init__(self, payload):
        self.structured_content = payload


class _Session:
    """No stored version to restore, HEAD's fit summary, and an identifiable fit's diagnosis."""

    def __init__(self, fit_summary):
        self.fit_summary = fit_summary

    async def call_tool(self, name, args):
        if name == "list_versions":
            return _FakeResult({"result": []})
        if name == "get_fit_summary":
            return _FakeResult(self.fit_summary)
        if name == "get_fit_diagnosis":
            ident = {"available": True, "fim": {"rank_deficient": False}, "flat_directions": []}
            return _FakeResult({"per_observable": [], "identifiability": ident})
        raise AssertionError(f"unexpected tool {name}")


#: A fit against GIVEN noise that fails chi2, as the server sends it with signals on.
_FAILS_CHI2 = {
    "neg_log_likelihood": 10.0,
    "bic": 25.0,
    "criterion_value": 25.0,
    "version": 3,
    "noise_sigma_report": {},
    "chi2_gof_pvalue": 1e-9,
}


def _finish(fit_summary, calls=1):
    from gemot import agent

    state: dict = {}
    tool = agent._finish_tool(_Session(fit_summary), state, adequacy="chi2")
    replies = [asyncio.run(tool.ainvoke({"rationale": "done"})) for _ in range(calls)]
    return replies, state


@pytest.mark.parametrize("signals", [True, False])
def test_off_the_chi2_gate_quotes_no_p_value_and_refuses_nothing(monkeypatch, signals):
    """Off, the server sends no p-value; the gate does not refuse on one, nor say it is missing."""
    pytest.importorskip("langchain_core", reason="harness-only dependency")
    from gemot import prompts

    monkeypatch.setattr(prompts, "PROGRESS_SIGNALS_ENABLED", signals)
    (reply,), state = _finish(_FAILS_CHI2)
    assert reply.startswith("FINISH REFUSED") is signals
    assert bool(state["refusals"]) is signals
    no_p = {k: v for k, v in _FAILS_CHI2.items() if k != "chi2_gof_pvalue"}
    (_, confirmed), _ = _finish(no_p, calls=2)
    assert confirmed.startswith("Recorded: finish")
    assert ("χ²" in confirmed) is signals


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
