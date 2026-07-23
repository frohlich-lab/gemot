"""Tests for the lead's finish gate (examples/_harness/agent.py `_decision_tool`).

The gate must: refuse a finish when there is no valid fit; refuse when the working model has
edits since its last fit (stale metrics); and otherwise require TWO finish calls — the first
ARMS and presents the residual diagnosis, the second CONFIRMS — re-arming if the model changed
in between. Driven against a fake MCP session (canned get_fit_summary / get_fit_diagnosis
payloads); langchain_core is a harness dep, so skip where it is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

_AGENT_PY = Path(__file__).resolve().parents[1] / "examples" / "_harness" / "agent.py"


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("_harness_agent_finish_gate", _AGENT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, payload):
        self.structuredContent = payload


class _FakeSession:
    """Serves canned get_fit_summary / get_fit_diagnosis payloads to the decision tool."""

    def __init__(self, fit_summary, diagnosis=None):
        self.fit_summary = fit_summary
        self.diagnosis = diagnosis or {"residual_structure": {"summary": "looks fine",
                                                              "flagged_observables": [],
                                                              "low_power": False},
                                       "per_observable": []}

    async def call_tool(self, name, args):
        if name == "get_fit_summary":
            return _FakeResult(self.fit_summary)
        if name == "get_fit_diagnosis":
            return _FakeResult(self.diagnosis)
        raise AssertionError(f"unexpected tool {name}")


async def _decide(tool, action):
    return await tool.ainvoke({"action": action, "rationale": "test"})


def _tool_for(mod, fit_summary, diagnosis=None):
    return mod._decision_tool(_FakeSession(fit_summary, diagnosis))


@pytest.mark.asyncio
async def test_finish_refused_without_valid_fit():
    mod = _load_agent_module()
    tool = _tool_for(mod, {"neg_log_likelihood": None, "bic": None, "version": 0})
    out = await _decide(tool, "finish")
    assert "FINISH REFUSED" in out and "no valid fit" in out


@pytest.mark.asyncio
async def test_finish_refused_with_unfit_edits():
    mod = _load_agent_module()
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 3,
                           "has_unfit_edits": True})
    out = await _decide(tool, "finish")
    assert "FINISH REFUSED" in out and "since the last fit" in out.lower()
    assert "call fit" in out.lower()


@pytest.mark.asyncio
async def test_finish_arms_then_confirms():
    mod = _load_agent_module()
    diag = {"residual_structure": {"summary": "systematic residual structure in obs_A",
                                   "flagged_observables": ["obs_A"], "low_power": False},
            "per_observable": [{"observable_id": "obs_A", "mean_weighted_residual": 2.0,
                                "sign_test": {"p": 0.002}, "runs_test": {"p": 0.3}}]}
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                           "has_unfit_edits": False}, diag)
    first = await _decide(tool, "finish")
    assert "FINISH ARMED" in first
    assert "systematic residual structure in obs_A" in first  # diagnosis shown
    assert "obs_A" in first and "0.002" in first               # worst-observable line
    second = await _decide(tool, "finish")
    assert second.startswith("Recorded: finish") and "bic=210" in second


@pytest.mark.asyncio
async def test_finish_rearms_when_version_changes():
    mod = _load_agent_module()
    session = _FakeSession({"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                            "has_unfit_edits": False})
    tool = mod._decision_tool(session)
    assert "FINISH ARMED" in await _decide(tool, "finish")
    # The agent edits + refits between the two finish calls -> new version -> must re-arm.
    session.fit_summary = {"neg_log_likelihood": 90.0, "bic": 205.0, "version": 6,
                           "has_unfit_edits": False}
    assert "FINISH ARMED" in await _decide(tool, "finish")   # NOT confirmed
    assert (await _decide(tool, "finish")).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_finish_arms_even_when_version_is_none():
    """After a rollback/restore_version, get_fit_summary reports version=None (the persisted
    sim blob carries no version). The gate must STILL arm on the first finish and confirm on the
    second — keying on version alone would make `None != None` False and confirm immediately,
    bypassing the gate on the common rollback->finish path."""
    mod = _load_agent_module()
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": None,
                           "has_unfit_edits": False})
    first = await _decide(tool, "finish")
    assert "FINISH ARMED" in first                              # NOT confirmed on first call
    second = await _decide(tool, "finish")
    assert second.startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_rollback_disarms_a_pending_finish():
    mod = _load_agent_module()
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                           "has_unfit_edits": False})
    assert "FINISH ARMED" in await _decide(tool, "finish")
    await _decide(tool, "rollback")                          # disarms
    assert "FINISH ARMED" in await _decide(tool, "finish")   # must arm again, not confirm


# --- _model_committed: the driver-loop guard for a silent stop that built nothing ------------
def _probe_returning(payload):
    async def probe(tool, args):
        assert tool == "describe_problem"
        return _FakeResult(payload)
    return probe


@pytest.mark.asyncio
async def test_model_committed_true_when_model_initialized():
    mod = _load_agent_module()
    assert await mod._model_committed(_probe_returning({"model_initialized": True})) is True


@pytest.mark.asyncio
async def test_model_committed_false_when_no_model():
    """The no-op case: a run that authored nothing (the lead stopped after only delegating to
    the biocurator) must be detected so the driver can nudge + resume instead of scoring k=0."""
    mod = _load_agent_module()
    assert await mod._model_committed(_probe_returning({"model_initialized": False})) is False


@pytest.mark.asyncio
async def test_model_committed_fails_open():
    """Must never nuke a real run: an absent probe, a raising probe, or a malformed payload all
    return True (assume a model exists) rather than triggering a spurious no-model resume."""
    mod = _load_agent_module()
    assert await mod._model_committed(None) is True

    async def raising(tool, args):
        raise RuntimeError("probe down")
    assert await mod._model_committed(raising) is True

    assert await mod._model_committed(_probe_returning({})) is True          # key missing
    assert await mod._model_committed(_probe_returning(None)) is True        # no structured content


def test_no_model_nudge_text_is_actionable():
    mod = _load_agent_module()
    assert "set_model" in mod._NO_MODEL_NUDGE and "fit" in mod._NO_MODEL_NUDGE.lower()
