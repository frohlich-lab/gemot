"""Tests for the lead's finish gate (petab_harness.agent `_finish_tool`).

The gate must: refuse a finish when there is no valid fit; refuse when the working model has
edits since its last fit (stale metrics); and otherwise require TWO finish calls — the first
ARMS and presents the residual diagnosis, the second CONFIRMS — re-arming if the model changed
in between. Driven against a fake MCP session (canned get_fit_summary / get_fit_diagnosis
payloads — the diagnosis carries BOTH sections, its parameter geometry under `identifiability`);
langchain_core is a harness dep, so skip where it is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

from petab_harness import agent as _agent  # noqa: E402


class _FakeResult:
    def __init__(self, payload):
        self.structuredContent = payload


# An identifiability payload the PARSIMONY gate reads as CLEAN (identifiable): available, a full-
# rank FIM, no flat directions. The default so existing tests exercise the gate's fail-through,
# not its refusal. `_ident(...)` builds the non-identifiable variants the parsimony tests need.
_IDENTIFIABLE = {"available": True, "method": "fim+multistart",
                 "fim": {"rank": 3, "n_estimated": 3, "condition": 12.0,
                         "rank_deficient": False},
                 "flat_directions": []}


def _ident(*, flat_directions=None, rank_deficient=False, available=True, fim=True):
    """The `identifiability` SECTION of a get_fit_diagnosis payload, for the parsimony gate.
    ``fim=False`` / ``available=False`` build the FAIL-OPEN shapes (no FIM was computable)."""
    return {
        "available": available,
        "method": "fim+multistart" if fim else "multistart",
        "fim": ({"rank": 2, "n_estimated": 3, "condition": 1e12,
                 "rank_deficient": rank_deficient} if fim else None),
        "flat_directions": flat_directions or [],
    }


class _FakeSession:
    """Serves canned get_fit_summary / get_fit_diagnosis payloads, and RECORDS the calls.

    The rolled-up residual verdict lives on the fit summary (the server reports it nowhere else) and
    the parameter geometry is a section of the one diagnosis call, so `identifiability` is spliced
    into the diagnosis payload rather than served as a second tool. Any other tool name is an
    assertion failure: the gate must not reach for one that no longer exists."""

    def __init__(self, fit_summary, diagnosis=None, identifiability=None):
        self.fit_summary = fit_summary
        self.diagnosis = diagnosis or {"per_observable": []}
        # Default CLEAN so a test that doesn't care about parsimony is unaffected by the gate.
        self.identifiability = identifiability if identifiability is not None else _IDENTIFIABLE
        self.calls: list[str] = []

    async def call_tool(self, name, args):
        self.calls.append(name)
        if name == "get_fit_summary":
            return _FakeResult(self.fit_summary)
        if name == "get_fit_diagnosis":
            return _FakeResult({**self.diagnosis, "identifiability": self.identifiability})
        raise AssertionError(f"unexpected tool {name}")


async def _finish(tool):
    return await tool.ainvoke({"rationale": "test"})


def _tool_for(mod, fit_summary, diagnosis=None, finish_state=None, identifiability=None):
    return mod._finish_tool(
        _FakeSession(fit_summary, diagnosis, identifiability), finish_state)


@pytest.mark.asyncio
async def test_finish_refused_without_valid_fit():
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": None, "bic": None, "version": 0})
    out = await _finish(tool)
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
        out = await _finish(tool)
        assert "FINISH REFUSED" in out and not out.startswith("Recorded: finish")
        assert fs.get("gave_up") is not True                 # not yet
    # The Nth attempt gives up — ends as no-model-built, still not an accepted finish.
    out = await _finish(tool)
    assert not out.startswith("Recorded: finish")
    assert "no-model-built" in out.lower()
    assert fs["gave_up"] is True


@pytest.mark.asyncio
async def test_no_fit_counter_is_consecutive_and_resets_on_a_fit():
    """The give-up counter is for CONSECUTIVE no-fit attempts: a valid fit appearing resets it."""
    mod = _agent
    session = _FakeSession({"neg_log_likelihood": None, "bic": None, "version": 0})
    tool = mod._finish_tool(session)
    for _ in range(mod._MAX_NO_FIT_FINISH_ATTEMPTS - 1):
        assert "FINISH REFUSED" in await _finish(tool)
    # A valid fit appears -> the next finish ARMS (counter reset), it does not give up.
    session.fit_summary = {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 1,
                           "has_unfit_edits": False}
    assert "FINISH ARMED" in await _finish(tool)
    # Fit gone again -> counting restarts from 1, so it refuses rather than immediately gives up.
    session.fit_summary = {"neg_log_likelihood": None, "bic": None, "version": 0}
    assert "FINISH REFUSED" in await _finish(tool)


@pytest.mark.asyncio
async def test_confirmed_finish_records_confirmed_flag():
    """A confirmed finish sets finish_state['confirmed'] so the driver won't bounce it back."""
    mod = _agent
    fs = {}
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                           "has_unfit_edits": False}, finish_state=fs)
    await _finish(tool)           # arm
    assert fs.get("confirmed") is not True   # not yet — a single call never confirms
    second = await _finish(tool)   # confirm
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
    out = await _finish(tool)
    assert "FINISH ARMED" in out
    assert "call fit" not in out.lower()


@pytest.mark.asyncio
async def test_finish_arms_then_confirms():
    mod = _agent
    diag = {"per_observable": [{"observable_id": "obs_A", "mean_weighted_residual": 2.0,
                                "sign_test": {"p": 0.002}, "runs_test": {"p": 0.3}}]}
    # The rolled-up verdict rides on the FIT — the diagnosis does not repeat it — so the armed
    # message has to read the summary from there and the worst-observable row from the diagnosis.
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                           "has_unfit_edits": False,
                           "residual_structure": {
                               "summary": "systematic residual structure in obs_A",
                               "flagged_observables": ["obs_A"], "low_power": False}}, diag)
    first = await _finish(tool)
    assert "FINISH ARMED" in first
    assert "systematic residual structure in obs_A" in first  # verdict off the fit result
    assert "obs_A" in first and "0.002" in first               # worst-observable line
    second = await _finish(tool)
    assert second.startswith("Recorded: finish") and "bic=210" in second


@pytest.mark.asyncio
async def test_finish_rearms_when_version_changes():
    mod = _agent
    session = _FakeSession({"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5,
                            "has_unfit_edits": False})
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    # The agent edits + refits between the two finish calls -> new version -> must re-arm.
    session.fit_summary = {"neg_log_likelihood": 90.0, "bic": 205.0, "version": 6,
                           "has_unfit_edits": False}
    assert "FINISH ARMED" in await _finish(tool)   # NOT confirmed
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_finish_arms_even_when_version_is_none():
    """A restored payload reports version=None (the persisted sim blob carries no version). The
    gate must STILL arm on the first finish and confirm on the second — keying on version alone
    would make `None != None` False and confirm immediately, bypassing the gate."""
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": None,
                           "has_unfit_edits": False})
    first = await _finish(tool)
    assert "FINISH ARMED" in first                              # NOT confirmed on first call
    second = await _finish(tool)
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
        out = await _finish(tool)
        assert "FINISH REFUSED" in out and "implausible noise model" in out
        assert "inflated" in out and "fix the mechanism" in out
        assert not out.startswith("Recorded: finish")
    # Bounded: the next attempt relents and ARMS instead of refusing forever.
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_collapsed_sigma_is_also_refused():
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_collapsed=True,
                               noise_sigma_advice="prefer a simpler mechanism."))
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "collapsed" in out


@pytest.mark.asyncio
async def test_refusal_names_both_tails_when_both_present():
    """With one observable inflated and another collapsed, the refusal names BOTH tails so the agent
    doesn't fix one and get silently re-blocked on the other."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=True, noise_sigma_collapsed=True,
                               noise_sigma_advice="fix the mechanism."))
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "inflated and collapsed" in out


@pytest.mark.asyncio
async def test_sigma_block_clears_when_the_fix_makes_sigma_plausible():
    """The refusal counter resets the moment the sigma is plausible again, so a real structural
    fix immediately clears the block (arms normally) rather than staying penalised."""
    mod = _agent
    session = _FakeSession(_fit(noise_sigma_inflated=True, noise_sigma_advice="x"))
    tool = mod._finish_tool(session)
    assert "FINISH REFUSED" in await _finish(tool)
    # Agent fixes the model FORM -> plausible sigma -> finish arms normally (counter reset).
    session.fit_summary = _fit(noise_sigma_inflated=False, noise_sigma_collapsed=False)
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_plausible_sigma_finish_is_unaffected():
    """A fit with no implausible-sigma flags arms/confirms exactly as before (no regression)."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=False, noise_sigma_collapsed=False))
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")


# --- the REPLICATE LACK-OF-FIT gate -----------------------------------------------------------
def _lof(*, available=True, n_flagged=1, n_points=15, status="flagged"):
    """A get_fit_summary residual_structure carrying the replicate arm's verdict, in the shape
    data_access._lof_reported puts on a fit result."""
    return {"residual_structure": {
        "summary": "systematic residual structure", "flagged_observables": ["observable_k5"],
        "low_power": False,
        "replicate_lack_of_fit": {
            "available": available, "status": status, "n_points": n_points, "alpha": 0.01,
            "n_flagged": n_flagged, "max_abs_z": 48.847, "min_p_holm": 1.51e-18,
            "points": [{"label": "observable_k5", "n": 18, "z": -48.847, "p_holm": 1.51e-18}]}}}


@pytest.mark.asyncio
async def test_finish_refused_on_replicate_lack_of_fit_then_relents():
    """The gate that carries what the deleted sigma-RATIO arm used to: blasi's dBIC +437 run sits
    under its data-std cap with a full-rank FIM, so neither the sigma nor the parsimony gate sees
    it, while the replicate arm condemns it at Holm p 1.5e-18. Bounded like the others."""
    mod = _agent
    tool = _tool_for(mod, _fit(**_lof()))
    for _ in range(mod._MAX_STRUCTURE_FINISH_REFUSALS):
        out = await _finish(tool)
        assert "FINISH REFUSED" in out and "REPLICATES rule this model out" in out
        assert "observable_k5" in out                 # names WHERE the form fails
        assert not out.startswith("Recorded: finish")
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_replicate_lack_of_fit_block_clears_on_a_structural_fix():
    mod = _agent
    session = _FakeSession(_fit(**_lof()))
    tool = mod._finish_tool(session)
    assert "FINISH REFUSED" in await _finish(tool)
    session.fit_summary = _fit(**_lof(n_flagged=0, status="pass"))
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_an_unavailable_replicate_verdict_never_refuses():
    """FAILS OPEN on the arm's own availability: a vacuous (over-parameterised), too-thin or
    untestable verdict reports NO EVIDENCE, which must not read as a condemnation. The
    over-parameterised case is the parsimony gate's, not this one's."""
    mod = _agent
    for verdict in (_lof(available=False, status="vacuous", n_flagged=0),
                    _lof(available=False, status="replicates_too_thin", n_flagged=0),
                    {"residual_structure": {"summary": "s", "flagged_observables": []}},  # no arm
                    {}):                                                    # no verdict at all
        assert "FINISH ARMED" in await _finish(_tool_for(mod, _fit(**verdict)))


@pytest.mark.asyncio
async def test_sign_runs_flags_alone_do_not_refuse():
    """Only the POWERED replicate arm gates. The sign/runs tests share flagged_observables with it
    but are advisory by design (low power at these sample sizes), so a flag from them alone must
    still arm — gating on it would refuse far beyond what this gate replaces."""
    mod = _agent
    sign_runs_only = {"residual_structure": {
        "summary": "sign-test bias in observable_k5", "flagged_observables": ["observable_k5"],
        "low_power": False,
        "replicate_lack_of_fit": {"available": False, "status": "no_replicated_points",
                                  "n_points": 0, "alpha": 0.01, "n_flagged": 0, "points": []}}}
    assert "FINISH ARMED" in await _finish(_tool_for(mod, _fit(**sign_runs_only)))


# --- the PARSIMONY / identifiability gate ----------------------------------------------------
_FLAT_16_ICS = _ident(
    rank_deficient=True,
    flat_directions=[{"eigenvalue": 1e-9, "rel": 1e-8,
                      "participants": [{"id": f"ic_{i}", "weight": 0.9} for i in range(4)]}
                     for _ in range(2)],
)


@pytest.mark.asyncio
async def test_finish_refused_on_non_identifiable_then_relents():
    """A finish on a fit whose FIM is rank-deficient / has flat directions (over-parameterised,
    the parameters unconstrained by the data) is REFUSED with GENERAL simplification advice — but
    bounded: after _MAX_PARSIMONY_FINISH_REFUSALS it relents (arms) so a model that genuinely
    can't be made identifiable still terminates at its honest score. Mirrors the sigma gate."""
    mod = _agent
    tool = _tool_for(mod, _fit(), identifiability=_FLAT_16_ICS)
    for _ in range(mod._MAX_PARSIMONY_FINISH_REFUSALS):
        out = await _finish(tool)
        assert "FINISH REFUSED" in out and "cannot constrain" in out
        assert "rank-deficient" in out.lower()
        # the flat directions are rendered as id·weight groups, not a flat name list
        assert "ic_0·0.9" in out
        assert "simplify" in out.lower()                          # GENERAL fix menu, not the answer
        assert "initial condition" not in out.lower()             # never ships the specific fix
        assert not out.startswith("Recorded: finish")
    # Bounded: the next attempt relents and ARMS instead of refusing forever.
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_rank_deficient_alone_refuses_even_without_named_flat_directions():
    """`rank_deficient` True is sufficient to refuse; with no bounded flat-direction participants
    to name, the refusal falls back to a generic phrasing rather than crashing."""
    mod = _agent
    tool = _tool_for(mod, _fit(), identifiability=_ident(rank_deficient=True, flat_directions=[]))
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "rank-deficient" in out.lower()
    assert "get_fit_diagnosis" in out                         # points to the full diagnosis


@pytest.mark.asyncio
async def test_flat_direction_without_rank_deficiency_does_NOT_refuse():
    """The calibration guard against over-refusal. A full-rank FIM that merely carries a near-flat
    eigen-direction (ill-conditioning: a parameter at a bound, or a correlated-but-identifiable
    group) is NOT refused — this is the enzyme_kinetics reference structure (rank 5/5, two flat
    directions, correct model). Only RANK DEFICIENCY triggers the gate; the finish arms normally."""
    mod = _agent
    ident = _ident(rank_deficient=False,
                   flat_directions=[{"eigenvalue": 1e-7, "rel": 1e-7,
                                     "participants": [{"id": "Km", "weight": 0.6},
                                                      {"id": "Ki", "weight": 0.6},
                                                      {"id": "Vmax", "weight": 0.5}]}])
    tool = _tool_for(mod, _fit(), identifiability=ident)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_parsimony_gate_fails_open_when_identifiability_unavailable():
    """FAIL OPEN: an unavailable diagnosis (available False, no FIM, or the probe returning {})
    must NOT refuse — identifiability is best-effort and cannot hold a finish hostage. The finish
    arms/confirms normally."""
    mod = _agent
    for ident in (_ident(available=False), _ident(fim=False), {}):
        tool = _tool_for(mod, _fit(), identifiability=ident)
        assert "FINISH ARMED" in await _finish(tool)
        assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_identifiable_fit_resets_the_parsimony_counter():
    """An identifiable fit RESETS the refusal counter, so a real simplification immediately clears
    the block (arms normally) rather than staying penalised — and a clean payload arms/confirms."""
    mod = _agent
    session = _FakeSession(_fit(), identifiability=_FLAT_16_ICS)
    tool = mod._finish_tool(session)
    assert "FINISH REFUSED" in await _finish(tool)     # refusal #1
    # Agent simplifies the FORM -> FIM now full-rank, no flat directions -> counter resets, arms.
    session.identifiability = _IDENTIFIABLE
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_clean_identifiable_finish_is_unaffected():
    """A fit with a full-rank FIM and no flat directions arms/confirms exactly as before."""
    mod = _agent
    tool = _tool_for(mod, _fit(), identifiability=_IDENTIFIABLE)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_one_diagnosis_call_serves_both_the_parsimony_gate_and_the_arm():
    """The residual detail and the FIM are two sections of ONE tool, so an armed finish costs one
    diagnosis probe, not two — and the gate must never reach for the deleted second tool (the fake
    raises on any other name)."""
    mod = _agent
    session = _FakeSession(_fit())
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    assert session.calls.count("get_fit_diagnosis") == 1
    assert set(session.calls) == {"get_fit_summary", "get_fit_diagnosis"}


#: A `merge_hint` as the server puts it on a fit result AND on get_fit_summary.
_HINT = {"parameters": ["ac16", "ac8"], "dnll": 0.001, "dbic": -5.528, "move_scaled": 0.002}


@pytest.mark.asyncio
async def test_the_armed_finish_names_an_outstanding_merge():
    """The tie is read off the SCORE, which is the gate's only view of HEAD: `get_fit_summary` now
    carries the same `merge_hint` the fit result does, so the gate names the pair the lead was
    already shown instead of projecting a second one out of the diagnosis (which needed
    `petab_mcp` imported into an import-light module to do)."""
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit(merge_hint=_HINT)))
    assert "FINISH ARMED" in armed and "OUTSTANDING MERGE" in armed
    assert "ac16 == ac8" in armed and "5.528" in armed        # ...and stated as a bic DECREASE
    assert "lowers bic" in armed.lower() and "-5.528" not in armed
    assert "0.002" in armed                                   # the move the panel must sanity-check
    # a clean fit says nothing: the line is absent, not "no merge found"
    assert "OUTSTANDING MERGE" not in await _finish(_tool_for(mod, _fit()))


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", [
    # REGRESSION: `abs(hint["dbic"] or 0.0)` — a tie beating the ln(n)/2 price by under 5e-4 nll
    # rounds to -0.0, which is FALSY, so the arm printed "LOWERS bic by 0.0" while telling the lead
    # to take the merge. A merge whose stated payoff is zero is not an argument.
    {**_HINT, "dbic": -0.0},
    {**_HINT, "dbic": None},          # ...and a genuinely absent dbic was swallowed the same way
])
async def test_the_armed_merge_never_advertises_a_zero_payoff(hint):
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit(merge_hint=hint)))
    assert "OUTSTANDING MERGE" in armed
    assert "lowers bic by 0.0" not in armed.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["not-a-dict", {}, {"parameters": []}, {"parameters": ["a", "b"]}])
async def test_a_malformed_merge_hint_never_raises_a_finish(hint):
    """REGRESSION: the merge rendering was the ONE step in the gate with no fail-open path (bare
    `hint[...]` subscripts, plus a lazy `from petab_mcp...` import in a module that must not depend
    on the server package). Every armed finish raised there, and the driver bounces a finish that
    did not confirm — so the run looped to the recursion limit instead of finishing at its honest
    score. A missing or odd hint means "no merge to report"."""
    mod = _agent
    state = {}
    tool = _tool_for(mod, _fit(merge_hint=hint), finish_state=state)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")
    assert state["confirmed"] is True


@pytest.mark.asyncio
async def test_a_stale_diagnosis_never_refuses_a_finish():
    """The server labels a diagnosis of an EARLIER fit `stale` rather than re-pricing the working
    model (which would mean building it). Refusing on it would refuse this finish over another
    structure's rank, so the parsimony gate fails open."""
    mod = _agent
    ident = {**_ident(rank_deficient=True, flat_directions=[{"participants": [{"id": "a"}]}]),
             "stale": True}
    assert "FINISH ARMED" in await _finish(_tool_for(mod, _fit(), identifiability=ident))


@pytest.mark.asyncio
async def test_parsimony_and_sigma_gates_compose():
    """The sigma gate runs FIRST: an implausible sigma is refused before the (expensive) FIM probe
    is even consulted, so a model with both problems is not double-charged in one call."""
    mod = _agent
    tool = _tool_for(mod, _fit(noise_sigma_inflated=True, noise_sigma_advice="fix the mechanism."),
                     identifiability=_FLAT_16_ICS)
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "implausible noise model" in out   # sigma wins this call


@pytest.mark.asyncio
async def test_a_refusal_disarms_a_pending_finish():
    """A refusal DISARMS: the next finish must re-present the diagnosis, not confirm against the
    stale one. The identifiability probe is what makes this reachable on a single fit — it is
    re-read live on every call, so a FIM that is unavailable/full-rank at arm time and
    rank-deficient on the next call refuses WHILE the fit's identity (bic, nll, version) is
    unchanged. Without the disarm the third call would confirm on a diagnosis shown two calls
    ago, which is exactly the review the arm->confirm handshake exists to force."""
    mod = _agent
    session = _FakeSession(_fit(), identifiability=_IDENTIFIABLE)
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    session.identifiability = _FLAT_16_ICS                     # refuses -> must disarm
    assert "FINISH REFUSED" in await _finish(tool)
    session.identifiability = _IDENTIFIABLE
    assert "FINISH ARMED" in await _finish(tool)               # re-arms, does NOT confirm
    assert (await _finish(tool)).startswith("Recorded: finish")


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


# --- the invalid-comparison note on the ARM (petab_mcp.nesting) ----------------------------
# The failure this addresses is real: blasi run 80ccf20efcbc020dc2becda41d14c62e, where the panel's
# ONLY "does adding structure help?" experiment (v3 -> v4) was a botched non-additive edit — k rose
# 6 -> 7 structural while the nll got 3.984 WORSE, because the new c816 factor REPLACED the c81216
# already in `r5_12_16_8` — and the lead read the worse score as evidence the model was complete,
# then finished at 4 fitted ops on 5% of its budget. The server flags that at commit time and keeps
# the flag until a valid additive experiment supersedes it; the gate must put it in front of the lead before it
# is allowed to confirm. It is a NOTE, not a refusal: exploring a non-nested variant is legitimate,
# and the run may have other grounds to finish.
_BLASI_ADVISORY = (
    "INVALID COMPARISON vs v3: k rose 7->8 but neg_log_likelihood got WORSE by 3.98381 — a model "
    "nested in its parent cannot fit worse. Either the edit was not additive (c81216 now appears "
    "in 8 places, was 9), or the fit under-converged."
)


@pytest.mark.asyncio
async def test_arm_surfaces_an_outstanding_invalid_comparison():
    mod = _agent
    tool = _tool_for(mod, _fit(invalid_comparison=_BLASI_ADVISORY))
    armed = await _finish(tool)
    assert "FINISH ARMED" in armed
    # the server's advisory is quoted verbatim, so the displaced parameter is named to the lead
    assert "c81216 now appears in 8 places, was 9" in armed
    # ...and the gate says what it means for the finish it is about to confirm
    assert "INVALID" in armed and "adding structure fails to help" in armed
    # NOTE, not refusal: a second finish still confirms.
    assert "Recorded: finish" in await _finish(tool)


@pytest.mark.asyncio
async def test_arm_is_silent_when_no_invalid_comparison_is_outstanding():
    """The key is ABSENT from get_fit_summary when the run has no outstanding invalid experiment
    (or the server predates the check), and the ARM text must then read exactly as before."""
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit()))
    assert "FINISH ARMED" in armed and "INVALID" not in armed
