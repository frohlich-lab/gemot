"""Tests for the lead's finish gate (petab_harness.agent `_decision_tool`).

The gate must: refuse a finish when there is no valid fit; refuse when the working model has
edits since its last fit (stale metrics); and otherwise require TWO finish calls — the first
ARMS and presents the residual diagnosis, the second CONFIRMS — re-arming if the model changed
in between. Driven against a fake MCP session (canned get_fit_summary / get_fit_diagnosis
payloads); langchain_core is a harness dep, so skip where it is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

from petab_harness import agent as _agent  # noqa: E402




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


def _tool_for(mod, fit_summary, diagnosis=None, finish_state=None):
    return mod._decision_tool(_FakeSession(fit_summary, diagnosis), finish_state)


@pytest.mark.asyncio
async def test_finish_refused_without_valid_fit():
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": None, "bic": None, "version": 0})
    out = await _decide(tool, "finish")
    assert "FINISH REFUSED" in out and "no valid fit" in out


@pytest.mark.asyncio
async def test_no_fit_finish_refuses_then_gives_up_never_accepts():
    """A finish with no valid fit is refused (never accepted as a finish), and after
    _MAX_NO_FIT_FINISH_ATTEMPTS consecutive tries the gate GIVES UP — flags the run for the
    driver to end as no-model-built, still NOT recording an accepted finish."""
    mod = _agent
    fs = {}
    tool = _tool_for(mod, {"neg_log_likelihood": None, "bic": None, "version": 0}, finish_state=fs)
    for _ in range(mod._MAX_NO_FIT_FINISH_ATTEMPTS - 1):
        out = await _decide(tool, "finish")
        assert "FINISH REFUSED" in out and not out.startswith("Recorded: finish")
        assert fs.get("gave_up") is not True                 # not yet
    # The Nth attempt gives up — ends as no-model-built, still not an accepted finish.
    out = await _decide(tool, "finish")
    assert not out.startswith("Recorded: finish")
    assert "no-model-built" in out.lower()
    assert fs["gave_up"] is True


@pytest.mark.asyncio
async def test_no_fit_counter_is_consecutive_and_resets_on_a_fit():
    """The give-up counter is for CONSECUTIVE no-fit attempts: a valid fit appearing resets it."""
    mod = _agent
    session = _FakeSession({"neg_log_likelihood": None, "bic": None, "version": 0})
    tool = mod._decision_tool(session)
    for _ in range(mod._MAX_NO_FIT_FINISH_ATTEMPTS - 1):
        assert "FINISH REFUSED" in await _decide(tool, "finish")
    # A valid fit appears -> the next finish ARMS (counter reset), it does not give up.
    session.fit_summary = {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 1,
                           "has_unfit_edits": False}
    assert "FINISH ARMED" in await _decide(tool, "finish")
    # Fit gone again -> counting restarts from 1, so it refuses rather than immediately gives up.
    session.fit_summary = {"neg_log_likelihood": None, "bic": None, "version": 0}
    assert "FINISH REFUSED" in await _decide(tool, "finish")


@pytest.mark.asyncio
async def test_confirmed_finish_records_confirmed_flag():
    """A confirmed finish sets finish_state['confirmed'] so the driver won't bounce it back."""
    mod = _agent
    fs = {}
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                           "has_unfit_edits": False}, finish_state=fs)
    await _decide(tool, "finish")           # arm
    assert fs.get("confirmed") is not True   # not yet — a single call never confirms
    second = await _decide(tool, "finish")   # confirm
    assert second.startswith("Recorded: finish")
    assert fs["confirmed"] is True


@pytest.mark.asyncio
async def test_finish_is_not_blocked_on_unfit_edits_because_there_cannot_be_any():
    """The gate used to refuse a finish whose metrics were stale — a model edited since its last
    fit. `make_changes` cannot leave that state (a change that fails to fit is reverted), so the
    branch is gone; had it stayed, its instruction ("call fit first") named a deleted tool and
    would have blocked the finish with no way out. The flag is still reported by
    describe_problem, so a real regression shows up there rather than as a wedged run."""
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 3,
                           "has_unfit_edits": True})
    out = await _decide(tool, "finish")
    assert "FINISH ARMED" in out
    assert "call fit" not in out.lower()


@pytest.mark.asyncio
async def test_finish_arms_then_confirms():
    mod = _agent
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
    mod = _agent
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
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": None,
                           "has_unfit_edits": False})
    first = await _decide(tool, "finish")
    assert "FINISH ARMED" in first                              # NOT confirmed on first call
    second = await _decide(tool, "finish")
    assert second.startswith("Recorded: finish")


def _fit(**kw):
    base = {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5, "has_unfit_edits": False}
    base.update(kw)
    return base


@pytest.mark.asyncio
async def test_finish_refused_on_implausible_sigma_then_relents():
    """A finish on a fit with an inflated (or collapsed) ESTIMATED sigma is REFUSED with the
    fix-the-mechanism advice — but bounded: after _MAX_SIGMA_FINISH_REFUSALS it relents (arms) so a
    model that genuinely can't reach a plausible sigma still terminates at its honest score."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=True,
                               noise_sigma_advice="fix the mechanism, not the noise."))
    for _ in range(mod._MAX_SIGMA_FINISH_REFUSALS):
        out = await _decide(tool, "finish")
        assert "FINISH REFUSED" in out and "implausible noise model" in out
        assert "inflated" in out and "fix the mechanism" in out
        assert not out.startswith("Recorded: finish")
    # Bounded: the next attempt relents and ARMS instead of refusing forever.
    assert "FINISH ARMED" in await _decide(tool, "finish")


@pytest.mark.asyncio
async def test_collapsed_sigma_is_also_refused():
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_collapsed=True,
                               noise_sigma_advice="prefer a simpler mechanism."))
    out = await _decide(tool, "finish")
    assert "FINISH REFUSED" in out and "collapsed" in out


@pytest.mark.asyncio
async def test_refusal_names_both_tails_when_both_present():
    """With one observable inflated and another collapsed, the refusal names BOTH tails so the agent
    doesn't fix one and get silently re-blocked on the other."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=True, noise_sigma_collapsed=True,
                               noise_sigma_advice="fix the mechanism."))
    out = await _decide(tool, "finish")
    assert "FINISH REFUSED" in out and "inflated and collapsed" in out


@pytest.mark.asyncio
async def test_sigma_block_clears_when_the_fix_makes_sigma_plausible():
    """The refusal counter resets the moment the sigma is plausible again, so a real structural
    fix immediately clears the block (arms normally) rather than staying penalised."""
    mod = _agent
    session = _FakeSession(_fit(noise_sigma_inflated=True, noise_sigma_advice="x"))
    tool = mod._decision_tool(session)
    assert "FINISH REFUSED" in await _decide(tool, "finish")
    # Agent fixes the model FORM -> plausible sigma -> finish arms normally (counter reset).
    session.fit_summary = _fit(noise_sigma_inflated=False, noise_sigma_collapsed=False)
    assert "FINISH ARMED" in await _decide(tool, "finish")


@pytest.mark.asyncio
async def test_plausible_sigma_finish_is_unaffected():
    """A fit with no implausible-sigma flags arms/confirms exactly as before (no regression)."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=False, noise_sigma_collapsed=False))
    assert "FINISH ARMED" in await _decide(tool, "finish")
    assert (await _decide(tool, "finish")).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_rollback_disarms_a_pending_finish():
    mod = _agent
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
async def test_model_committed_true_when_model_authored():
    mod = _agent
    assert await mod._model_committed(_probe_returning({"model_authored": True})) is True


@pytest.mark.asyncio
async def test_model_committed_false_when_no_model():
    """The no-op case: a run that authored nothing (the lead stopped after only delegating to
    the biocurator) must be detected so the driver can nudge + resume instead of scoring k=0."""
    mod = _agent
    assert await mod._model_committed(_probe_returning({"model_authored": False})) is False


@pytest.mark.asyncio
async def test_model_committed_fails_open():
    """Must never nuke a real run: an absent probe, a raising probe, or a malformed payload all
    return True (assume a model exists) rather than triggering a spurious no-model resume."""
    mod = _agent
    assert await mod._model_committed(None) is True

    async def raising(tool, args):
        raise RuntimeError("probe down")
    assert await mod._model_committed(raising) is True

    assert await mod._model_committed(_probe_returning({})) is True          # key missing
    assert await mod._model_committed(_probe_returning(None)) is True        # no structured content


def test_no_model_nudge_text_is_actionable():
    mod = _agent
    assert "make_changes" in mod._NO_MODEL_NUDGE and "fit" in mod._NO_MODEL_NUDGE.lower()
