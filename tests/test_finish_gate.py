"""Tests for the lead's finish gate (gemot.agent `_finish_tool`).

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

from gemot import agent as _agent  # noqa: E402
from gemot import prompts  # noqa: E402


class _FakeResult:
    def __init__(self, payload):
        self.structured_content = payload


# An identifiability payload the PARSIMONY gate reads as CLEAN (identifiable): available, a full-
# rank FIM, no flat directions. The default so existing tests exercise the gate's fail-through,
# not its refusal. `_ident(...)` builds the non-identifiable variants the parsimony tests need.
_IDENTIFIABLE = {
    "available": True,
    "method": "fim+multistart",
    "fim": {"rank": 3, "n_estimated": 3, "condition": 12.0, "rank_deficient": False},
    "flat_directions": [],
}


def _ident(*, flat_directions=None, rank_deficient=False, available=True, fim=True):
    """The `identifiability` SECTION of a get_fit_diagnosis payload, for the parsimony gate.
    ``fim=False`` / ``available=False`` build the FAIL-OPEN shapes (no FIM was computable)."""
    return {
        "available": available,
        "method": "fim+multistart" if fim else "multistart",
        "fim": (
            {"rank": 2, "n_estimated": 3, "condition": 1e12, "rank_deficient": rank_deficient}
            if fim
            else None
        ),
        "flat_directions": flat_directions or [],
    }


class _FakeSession:
    """Serves canned get_fit_summary / get_fit_diagnosis payloads, and RECORDS the calls.

    The rolled-up residual verdict lives on the fit summary (the server reports it nowhere else) and
    the parameter geometry is a section of the one diagnosis call, so `identifiability` is spliced
    into the diagnosis payload rather than served as a second tool. Any other tool name is an
    assertion failure: the gate must not reach for one that no longer exists.

    ``versions`` are the `list_versions` rows the gate ranks to find the version the run is scored
    on, ``metas`` the stored meta `get_version` returns for a version (its noise-sigma flags),
    ``head`` the version `describe_problem` reports, and ``summaries`` the fit summary each version
    restores (`restore_version` swaps it in and moves HEAD). With no versions, the gate has nothing
    to restore and judges HEAD.

    ``parameters`` are the `get_parameters` rows (id and role) and ``model_parameters`` the ids
    `get_model_structure` says the model declares, which the parsimony gate reads to leave out the
    flat directions only the fixed tables' parameters carry."""

    def __init__(
        self,
        fit_summary,
        diagnosis=None,
        identifiability=None,
        versions=(),
        head=None,
        summaries=None,
        metas=None,
        parameters=(),
        model_parameters=(),
    ):
        self.fit_summary = fit_summary
        self.diagnosis = diagnosis or {"per_observable": []}
        # Default CLEAN so a test that doesn't care about parsimony is unaffected by the gate.
        self.identifiability = identifiability if identifiability is not None else _IDENTIFIABLE
        self.versions = list(versions)
        self.head = head
        self.summaries = summaries or {}
        self.metas = metas or {}
        # JSON lists, as the server sends them; anything else as given, for the odd shapes.
        self.parameters = list(parameters) if isinstance(parameters, tuple) else parameters
        self.model_parameters = (
            list(model_parameters) if isinstance(model_parameters, tuple) else model_parameters
        )
        self.calls: list[str] = []
        self.restored: list[int] = []

    async def call_tool(self, name, args):
        self.calls.append(name)
        if name == "get_fit_summary":
            return _FakeResult(self.fit_summary)
        if name == "get_fit_diagnosis":
            return _FakeResult({**self.diagnosis, "identifiability": self.identifiability})
        if name == "list_versions":
            return _FakeResult({"result": self.versions})
        if name == "describe_problem":
            return _FakeResult({"current_version": self.head})
        if name == "get_version":
            return _FakeResult({"version": args["version"], **self.metas.get(args["version"], {})})
        if name == "get_parameters":
            return _FakeResult({"result": self.parameters})
        if name == "get_model_structure":
            return _FakeResult({"species": [], "parameters": self.model_parameters, "readouts": {}})
        if name == "restore_version":
            if args["version"] not in self.summaries:
                return _FakeResult({"ok": False, "error": "no such version"})
            self.restored.append(args["version"])
            self.head = args["version"]
            self.fit_summary = self.summaries[args["version"]]
            return _FakeResult({"version": args["version"]})
        raise AssertionError(f"unexpected tool {name}")


async def _finish(tool):
    return await tool.ainvoke({"rationale": "test"})


def _tool_for(mod, fit_summary, diagnosis=None, finish_state=None, identifiability=None):
    return mod._finish_tool(_FakeSession(fit_summary, diagnosis, identifiability), finish_state)


@pytest.mark.asyncio
async def test_finish_refused_without_valid_fit():
    mod = _agent
    tool = _tool_for(mod, {"neg_log_likelihood": None, "bic": None, "version": 0})
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "no valid fit" in out


@pytest.mark.asyncio
async def test_finish_needs_the_RANKED_score_not_merely_a_finite_bic():
    """The gate finishes on `criterion_value` — the score `record_fit_progress` checkpoints on —
    not on canonical bic. The two part exactly where the run has NO scored model: aicc is
    undefined at n <= k+1 and hqc below n = 3, and bic is finite at both. Gating on bic there
    would let a run finish on a fit the session never scored, leaving the judge no checkpoint."""
    mod = _agent
    refused = await _finish(_tool_for(mod, _fit(criterion_value=None)))
    assert "FINISH REFUSED" in refused and "no valid fit" in refused
    # ...and a finite ranked score arms as usual, even when it differs from bic.
    armed = await _finish(_tool_for(mod, _fit(criterion_value=205.0)))
    assert "FINISH ARMED" in armed


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
        assert fs.get("gave_up") is not True  # not yet
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
    session.fit_summary = {
        "neg_log_likelihood": 100.0,
        "bic": 210.0,
        "version": 1,
        "has_unfit_edits": False,
    }
    assert "FINISH ARMED" in await _finish(tool)
    # Fit gone again -> counting restarts from 1, so it refuses rather than immediately gives up.
    session.fit_summary = {"neg_log_likelihood": None, "bic": None, "version": 0}
    assert "FINISH REFUSED" in await _finish(tool)


@pytest.mark.asyncio
async def test_confirmed_finish_records_confirmed_flag():
    """A confirmed finish sets finish_state['confirmed'] so the driver won't bounce it back."""
    mod = _agent
    fs = {}
    tool = _tool_for(
        mod,
        {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5, "has_unfit_edits": False},
        finish_state=fs,
    )
    await _finish(tool)  # arm
    assert fs.get("confirmed") is not True  # not yet — a single call never confirms
    second = await _finish(tool)  # confirm
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
    tool = _tool_for(
        mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 3, "has_unfit_edits": True}
    )
    out = await _finish(tool)
    assert "FINISH ARMED" in out
    assert "call fit" not in out.lower()


@pytest.mark.asyncio
async def test_finish_arms_then_confirms():
    mod = _agent
    diag = {
        "per_observable": [
            {
                "observable_id": "obs_A",
                "mean_weighted_residual": 2.0,
                "sign_test": {"p": 0.002},
                "runs_test": {"p": 0.3},
            }
        ]
    }
    # The rolled-up verdict rides on the FIT — the diagnosis does not repeat it — so the armed
    # message has to read the summary from there and the worst-observable row from the diagnosis.
    tool = _tool_for(
        mod,
        {
            "neg_log_likelihood": 100.0,
            "bic": 210.0,
            "version": 5,
            "has_unfit_edits": False,
            "residual_structure": {
                "summary": "systematic residual structure in obs_A",
                "flagged_observables": ["obs_A"],
                "low_power": False,
            },
        },
        diag,
    )
    first = await _finish(tool)
    assert "FINISH ARMED" in first
    assert "systematic residual structure in obs_A" in first  # verdict off the fit result
    assert "obs_A" in first and "0.002" in first  # worst-observable line
    second = await _finish(tool)
    assert second.startswith("Recorded: finish") and "bic=210" in second


@pytest.mark.asyncio
async def test_finish_rearms_when_version_changes():
    mod = _agent
    session = _FakeSession(
        {"neg_log_likelihood": 100.0, "bic": 210.0, "version": 5, "has_unfit_edits": False}
    )
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    # The agent edits + refits between the two finish calls -> new version -> must re-arm.
    session.fit_summary = {
        "neg_log_likelihood": 90.0,
        "bic": 205.0,
        "version": 6,
        "has_unfit_edits": False,
    }
    assert "FINISH ARMED" in await _finish(tool)  # NOT confirmed
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_finish_arms_even_when_version_is_none():
    """A restored payload reports version=None (the persisted sim blob carries no version). The
    gate must STILL arm on the first finish and confirm on the second — keying on version alone
    would make `None != None` False and confirm immediately, bypassing the gate."""
    mod = _agent
    tool = _tool_for(
        mod, {"neg_log_likelihood": 100.0, "bic": 210.0, "version": None, "has_unfit_edits": False}
    )
    first = await _finish(tool)
    assert "FINISH ARMED" in first  # NOT confirmed on first call
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
    tool = _tool_for(
        mod, _fit(noise_sigma_inflated=True, noise_sigma_advice="fix the mechanism, not the noise.")
    )
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
    tool = _tool_for(
        mod, _fit(noise_sigma_collapsed=True, noise_sigma_advice="prefer a simpler mechanism.")
    )
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "collapsed" in out


@pytest.mark.asyncio
async def test_refusal_names_both_tails_when_both_present():
    """With one observable inflated and another collapsed, the refusal names BOTH tails so the agent
    doesn't fix one and get silently re-blocked on the other."""
    mod = _agent
    tool = _tool_for(
        mod,
        _fit(
            noise_sigma_inflated=True,
            noise_sigma_collapsed=True,
            noise_sigma_advice="fix the mechanism.",
        ),
    )
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


# --- residual flags never refuse ------------------------------------------------------------
def _lof(*, available=True, n_flagged=1, n_points=15, status="flagged"):
    """A get_fit_summary residual_structure carrying the replicate arm's verdict, in the shape
    data_access._lof_reported puts on a fit result."""
    return {
        "residual_structure": {
            "summary": "systematic residual structure",
            "flagged_observables": ["observable_k5"],
            "low_power": False,
            "replicate_lack_of_fit": {
                "available": available,
                "status": status,
                "n_points": n_points,
                "alpha": 0.01,
                "n_flagged": n_flagged,
                "max_abs_z": 48.847,
                "min_p_holm": 1.51e-18,
                "points": [{"label": "observable_k5", "n": 18, "z": -48.847, "p_holm": 1.51e-18}],
            },
        }
    }


_SIGN_RUNS_ONLY = {
    "residual_structure": {
        "summary": "sign-test bias in observable_k5",
        "flagged_observables": ["observable_k5"],
        "low_power": False,
        "replicate_lack_of_fit": {
            "available": False,
            "status": "no_replicated_points",
            "n_points": 0,
            "alpha": 0.01,
            "n_flagged": 0,
            "points": [],
        },
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [_lof(), _SIGN_RUNS_ONLY], ids=["replicate", "sign_runs"])
async def test_a_residual_flag_arms_and_confirms(verdict):
    """gemot#97: a replicate lack-of-fit flag used to refuse up to three times, and in sweep round 3
    it refused every lucarelli finish, the best run's (dBIC -545) included, while the flags at the
    scored version did not separate passing from failing runs on any problem with enough failures.
    No flag refuses now; the verdict is shown at arm time, where the lead reads it."""
    tool = _tool_for(_agent, _fit(**verdict))
    out = await _finish(tool)
    assert "FINISH ARMED" in out and "FINISH REFUSED" not in out
    assert verdict["residual_structure"]["summary"] in out
    assert (await _finish(tool)).startswith("Recorded: finish")


# The server's caveat on a clean summary where sigmas are estimated, verbatim from
# petab_mcp.data_access._ABSORBED_MISFIT_CAVEAT.
_CAVEAT = (
    "Estimated noise sigmas absorb misfit, so the sign/runs tests have little power here and no "
    "structure flag is NOT adequacy. get_fit_diagnosis's sigma_heterogeneity and "
    "noise_sigma.fitted_over_data_noise rank where to look."
)


def _sigma(pid, p, ratio=2.4):
    """One `residual_structure.sigma_heterogeneity` row, in the shape data_access._absorbed_misfit
    puts on get_fit_summary."""
    return {
        "parameter_id": pid,
        "p": p,
        "worst_series": f"{pid}_gel2_data3",
        "fitted_over_data_noise": ratio,
    }


@pytest.mark.asyncio
async def test_arm_text_carries_the_caveat_and_the_heterogeneous_sigmas():
    """gemot#111: on a problem with estimated sigmas (bachmann, fiedler) the ARM text quoted "no
    significant residual structure" alone, though the sign/runs tests have little power there and a
    sigma inflated for one series hides misfit from them. The server's caveat and the sigmas it names
    (PEtab-MCP#212) now follow the summary, the first three by p with the rest counted; the finish
    still arms and confirms, as no residual evidence refuses one (gemot#97)."""
    named = [
        _sigma("sigma_pMek", 0.0018),
        _sigma("sigma_pErk", 0.0031, ratio=None),
        _sigma("sigma_pRaf", 0.01),
        _sigma("sigma_pAkt", 0.02),
        _sigma("sigma_pS6", 0.03),
    ]
    rs = {
        "summary": "no significant residual structure (sign-test bias or runs-test time trend).",
        "flagged_observables": [],
        "low_power": False,
        "sigma_heterogeneity": named,
        "structure_tests_caveat": _CAVEAT,
    }
    tool = _tool_for(_agent, _fit(residual_structure=rs))
    out = await _finish(tool)
    assert "FINISH ARMED" in out and "FINISH REFUSED" not in out
    assert f"{rs['summary']} {_CAVEAT}" in out  # the caveat qualifies the summary it follows
    assert "SIGMA HETEROGENEITY" in out
    assert "sigma_pMek (p=0.0018, worst series sigma_pMek_gel2_data3, " in out
    assert "fitted_over_data_noise=2.4)" in out
    assert "sigma_pErk (p=0.0031, worst series sigma_pErk_gel2_data3, " in out
    assert "fitted_over_data_noise=unknown)" in out  # a sigma the cap report cannot price
    assert "sigma_pRaf" in out
    assert "sigma_pAkt" not in out and "sigma_pS6" not in out
    assert prompts.FINISH_ARMED_SIGMA_MORE_TMPL.format(n=2) in out
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_arm_text_adds_nothing_without_the_caveat_or_named_sigmas():
    """gemot#111's other half: a rollup that carries neither key (fixed sigmas, or a fit that flags
    an observable and names no sigma) arms on the summary exactly as before, with no empty label."""
    rs = {
        "summary": "systematic residual structure in obs_A.",
        "flagged_observables": ["obs_A"],
        "low_power": False,
    }
    out = await _finish(_tool_for(_agent, _fit(residual_structure=rs)))
    assert f"Residual structure: {rs['summary']} " + prompts._RESIDUALS_DO_NOT_DECIDE_DONE in out
    assert "SIGMA HETEROGENEITY" not in out and "absorb misfit" not in out


# --- the PARSIMONY / identifiability gate ----------------------------------------------------
_FLAT_16_ICS = _ident(
    rank_deficient=True,
    flat_directions=[
        {
            "eigenvalue": 1e-9,
            "rel": 1e-8,
            "participants": [{"id": f"ic_{i}", "weight": 0.9} for i in range(4)],
        }
        for _ in range(2)
    ],
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
        assert "simplify" in out.lower()  # GENERAL fix menu, not the answer
        assert "initial condition" not in out.lower()  # never ships the specific fix
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
    assert "get_fit_diagnosis" in out  # points to the full diagnosis


@pytest.mark.asyncio
async def test_flat_direction_without_rank_deficiency_does_NOT_refuse():
    """The calibration guard against over-refusal. A full-rank FIM that merely carries a near-flat
    eigen-direction (ill-conditioning: a parameter at a bound, or a correlated-but-identifiable
    group) is NOT refused — this is the enzyme_kinetics reference structure (rank 5/5, two flat
    directions, correct model). Only RANK DEFICIENCY triggers the gate; the finish arms normally."""
    mod = _agent
    ident = _ident(
        rank_deficient=False,
        flat_directions=[
            {
                "eigenvalue": 1e-7,
                "rel": 1e-7,
                "participants": [
                    {"id": "Km", "weight": 0.6},
                    {"id": "Ki", "weight": 0.6},
                    {"id": "Vmax", "weight": 0.5},
                ],
            }
        ],
    )
    tool = _tool_for(mod, _fit(), identifiability=ident)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")


@pytest.mark.asyncio
async def test_parsimony_gate_fails_open_when_identifiability_unavailable():
    """FAIL OPEN: an unavailable diagnosis (available False, no FIM, or the probe returning {})
    must NOT refuse — identifiability is best-effort and cannot hold a finish hostage. The finish
    arms/confirms normally."""
    mod = _agent
    no_rank = {**_IDENTIFIABLE, "fim": {"rank": 3, "n_estimated": 3}}
    note = prompts.FINISH_PARSIMONY_UNCHECKED_TMPL.format(why=prompts.FINISH_PARSIMONY_UNAVAILABLE)
    for ident in (_ident(available=False), _ident(fim=False), {}, no_rank):
        fs = {}
        tool = _tool_for(mod, _fit(), identifiability=ident, finish_state=fs)
        armed = await _finish(tool)
        assert "FINISH ARMED" in armed and armed.endswith(note)
        confirmed = await _finish(tool)
        assert confirmed.startswith("Recorded: finish") and confirmed.endswith(note)
        # gemot#162: each unchecked pass is a relent of its own, never a silent one
        assert fs["relents"] == {"parsimony_unavailable": 2} and fs["refusals"] == {}


@pytest.mark.asyncio
async def test_identifiable_fit_resets_the_parsimony_counter():
    """An identifiable fit RESETS the refusal counter, so a real simplification immediately clears
    the block (arms normally) rather than staying penalised — and a clean payload arms/confirms."""
    mod = _agent
    session = _FakeSession(_fit(), identifiability=_FLAT_16_ICS)
    tool = mod._finish_tool(session)
    assert "FINISH REFUSED" in await _finish(tool)  # refusal #1
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
    assert set(session.calls) == {"list_versions", "get_fit_summary", "get_fit_diagnosis"}


#: A `merge_hints` row as the server puts it on a fit result AND on get_fit_summary.
_HINT = {"parameters": ["ac16", "ac8"], "dnll": 0.001, "dscore": -5.528, "move_scaled": 0.002}
#: The other two rows of a redundant GROUP, as the server sends them cheapest-first.
_HINTS = [
    _HINT,
    {"parameters": ["ac16", "ac9"], "dnll": 0.004, "dscore": -5.522, "move_scaled": 0.01},
    {"parameters": ["dc1", "dc2"], "dnll": 0.02, "dscore": -5.49, "move_scaled": 0.3},
]


@pytest.mark.asyncio
async def test_the_armed_finish_names_a_priced_merge():
    """The ties are read off the SCORE, which is the gate's only view of HEAD: `get_fit_summary` now
    carries the same `merge_hints` the fit result does, so the gate names the pairs the lead was
    already shown instead of projecting new ones out of the diagnosis (which needed
    `petab_mcp` imported into an import-light module to do)."""
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit(merge_hints=[_HINT])))
    assert "FINISH ARMED" in armed and "PRICED MERGES" in armed
    assert "ac16 == ac8" in armed and "5.528" in armed  # ...and stated as a score DECREASE
    assert "lowers the score" in armed.lower() and "-5.528" not in armed
    assert "0.002" in armed  # the move the panel must sanity-check
    # a clean fit says nothing: the line is absent, not "no merge found"
    assert "PRICED MERGES" not in await _finish(_tool_for(mod, _fit()))


_PINS = [
    {"parameter_id": "k_offI", "side": "upper", "value": 99.99998, "bound_value": 100.0},
    {"parameter_id": "k1e", "side": "lower", "value": 1.0000001e-06, "bound_value": 1e-06},
]


@pytest.mark.asyncio
async def test_the_armed_finish_names_every_pinned_parameter_with_its_bound():
    """PEtab-MCP#334: with no pin line in the finish review, run 63612603_28 reported `k_offI` as
    interior at 3.28 while it sat at its upper bound of 100. The ARM text names every pin of the
    judged version off `get_fit_summary`'s full `parameters_at_bound` (a fit result splits it into
    new and held, PEtab-MCP#368), and a version with none says nothing."""
    armed = await _finish(_tool_for(_agent, _fit(parameters_at_bound=_PINS)))
    assert "FINISH ARMED" in armed and "PINNED" in armed
    assert "k_offI at its upper bound 100" in armed and "k1e at its lower bound 1e-06" in armed
    assert "PINNED" not in await _finish(_tool_for(_agent, _fit(parameters_at_bound=[])))
    assert "PINNED" not in await _finish(_tool_for(_agent, _fit()))


@pytest.mark.parametrize(
    "pins",
    [
        "k_offI",
        [None, 3, {"side": "upper", "bound_value": 1.0}, {"parameter_id": "k", "side": None}],
        [{"parameter_id": "k", "side": "upper", "bound_value": "inf"}, *_PINS],
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_pin_never_raises_a_finish(pins):
    """The line fails open like the merge line: an odd payload costs the line, and an unreadable
    row does not silence the readable ones."""
    armed = await _finish(_tool_for(_agent, _fit(parameters_at_bound=pins)))
    assert armed.startswith("FINISH ARMED")
    named = isinstance(pins, list) and _PINS[0] in pins
    assert ("k_offI at its upper bound 100" in armed) is named


@pytest.mark.asyncio
async def test_the_armed_finish_names_every_priced_tie_not_just_the_cheapest():
    """One named tie per arm is what taught the lead to merge PIECEMEAL — take the cheapest, refit,
    be shown the next member of the same redundant group, refit again. The arm names all of them, and
    licenses only the edits that were PRICED: independent pairs together in one call, an overlapping
    group one pair at a time.

    Both halves are load-bearing, and the second was learned the expensive way (blasi, the first run
    under the multi-hint text): "take them ALL in ONE call" plus "an overlapping group is one
    multi-way tie" produced a 3-symbol collapse worth +186 bic against it. The arm must not be able to
    read as a license for that again."""
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit(merge_hints=_HINTS)))
    for hint in _HINTS:
        assert " == ".join(hint["parameters"]) in armed
    # Against the CONSTANT, not a hand-copied phrase: the rule is one string composed into every
    # surface that states it (prompts._MERGE_RULE), so quoting a fragment of it here would be one
    # more copy to keep in step with the text.
    assert prompts._MERGE_RULE in armed
    # ...and it really is the rule this test is about: independent pairs together, an overlapping
    # group one pair at a time, the unpriced collapse named and refused — and every one of them an
    # option, never a condition for finishing (gemot#94: told a bottomed-out bic with merges
    # outstanding was "NOT a finished model", runs tied a closing rate to a Ca2+ midpoint).
    for claim in (
        "ONE `make_changes` call",
        "PRICED",
        "PAIR alone",
        "Do NOT collapse three symbols",
        "OPTION, not a condition for finishing",
    ):
        assert claim in prompts._MERGE_RULE, claim
    assert "multi-way tie" not in armed
    assert "  " not in armed  # the tie lines carry their own leading space


@pytest.mark.asyncio
async def test_a_broken_merge_template_costs_the_line_not_the_finish():
    """REGRESSION: rendering the ties moved the outer `.format` OUTSIDE the try that wraps the
    per-row one, so an unfilled slot in the surrounding template would raise out of the ARM branch
    — and the driver bounces a finish that did not confirm, so the run loops to the recursion limit
    instead of finishing at its honest score. Every step of the advisory fails open."""
    mod = _agent
    broken = "{ties}{a_slot_nobody_fills}"
    original = prompts.FINISH_ARMED_MERGE_TMPL
    prompts.FINISH_ARMED_MERGE_TMPL = broken
    try:
        state = {}
        tool = _tool_for(mod, _fit(merge_hints=_HINTS), finish_state=state)
        armed = await _finish(tool)
        assert "FINISH ARMED" in armed and "PRICED MERGES" not in armed
        assert (await _finish(tool)).startswith("Recorded: finish")
        assert state["confirmed"] is True
    finally:
        prompts.FINISH_ARMED_MERGE_TMPL = original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hint",
    [
        # REGRESSION: `abs(hint["dscore"] or 0.0)` — a tie beating the ln(n)/2 price by under 5e-4 nll
        # rounds to -0.0, which is FALSY, so the arm printed "LOWERS bic by 0.0" while telling the lead
        # to take the merge. A merge whose stated payoff is zero is not an argument.
        {**_HINT, "dscore": -0.0},
        {**_HINT, "dscore": None},  # ...and a genuinely absent dscore was swallowed the same way
    ],
)
async def test_the_armed_merge_never_advertises_a_zero_payoff(hint):
    mod = _agent
    armed = await _finish(_tool_for(mod, _fit(merge_hints=[hint])))
    assert "PRICED MERGES" in armed
    assert "lowers bic by 0.0" not in armed.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hints", ["not-a-list", {"parameters": ["a", "b"]}, [], [None], ["not-a-dict"], [{}]]
)
async def test_a_malformed_merge_hints_payload_never_raises_a_finish(hints):
    """The field itself, not just its rows: anything that is not a list of rows means "no merge to
    report". A bare dict is included because that is the shape this field used to have."""
    mod = _agent
    state = {}
    tool = _tool_for(mod, _fit(merge_hints=hints), finish_state=state)
    armed = await _finish(tool)
    assert "FINISH ARMED" in armed and "PRICED MERGES" not in armed
    assert (await _finish(tool)).startswith("Recorded: finish")
    assert state["confirmed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["not-a-dict", {}, {"parameters": []}, {"parameters": ["a", "b"]}])
async def test_a_malformed_merge_hint_row_never_raises_a_finish(hint):
    """REGRESSION: the merge rendering was the ONE step in the gate with no fail-open path (bare
    `hint[...]` subscripts, plus a lazy `from petab_mcp...` import in a module that must not depend
    on the server package). Every armed finish raised there, and the driver bounces a finish that
    did not confirm — so the run looped to the recursion limit instead of finishing at its honest
    score. A missing or odd hint means "no merge to report"."""
    mod = _agent
    state = {}
    tool = _tool_for(mod, _fit(merge_hints=[hint]), finish_state=state)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")
    assert state["confirmed"] is True


@pytest.mark.asyncio
async def test_a_stale_diagnosis_never_refuses_a_finish():
    """The server labels a diagnosis of an EARLIER fit `stale` rather than re-pricing the working
    model (which would mean building it). Refusing on it would refuse this finish over another
    structure's rank, so the parsimony gate fails open."""
    mod = _agent
    ident = {
        **_ident(rank_deficient=True, flat_directions=[{"participants": [{"id": "a"}]}]),
        "stale": True,
    }
    fs = {}
    armed = await _finish(_tool_for(mod, _fit(), identifiability=ident, finish_state=fs))
    assert "FINISH ARMED" in armed
    assert armed.endswith(
        prompts.FINISH_PARSIMONY_UNCHECKED_TMPL.format(why=prompts.FINISH_PARSIMONY_STALE)
    )
    assert fs["relents"] == {"parsimony_stale": 1}


def test_a_checked_diagnosis_is_not_unchecked():
    """The fail-open record must not fire on a diagnosis the gate did read, either way it read."""
    unchecked = _agent.finish_gate._parsimony_unchecked
    assert unchecked(_IDENTIFIABLE) is None
    assert unchecked(_FLAT_16_ICS) is None


# What the server answers after a restore when the restored version's fit stored no verdict of its
# own (froehlich 63619743_29 and lucarelli 63968832_7 stored no FIM, gemot#162): the in-memory fit
# record is another structure's, so the section is either that record labelled stale or no verdict.
_STALE_AFTER_RESTORE = {
    **_IDENTIFIABLE,
    "stale": True,
    "stale_note": "this describes the LAST FIT, not the model in front of you",
}
_NONE_AFTER_RESTORE = {"available": False, "method": "none", "note": "no fitted optimum"}


class _RestoreSession(_FakeSession):
    """A :class:`_FakeSession` whose diagnosis turns into ``after_restore`` once a version is
    restored, as the server's does when the restored version's own verdict is not on hand."""

    def __init__(self, *args, after_restore, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_restore = after_restore

    async def call_tool(self, name, args):
        res = await super().call_tool(name, args)
        if name == "restore_version" and args["version"] in self.restored:
            self.identifiability = self.after_restore
        return res


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after_restore", "reason", "why"),
    [
        (_STALE_AFTER_RESTORE, "stale", prompts.FINISH_PARSIMONY_STALE),
        (_NONE_AFTER_RESTORE, "unavailable", prompts.FINISH_PARSIMONY_UNAVAILABLE),
    ],
)
async def test_a_refused_version_restored_and_passed_unchecked_is_a_relent(
    after_restore, reason, why, capsys
):
    """REGRESSION gemot#162: refused on v15, the lead explores on to v20, and the next finish
    restores v15 onto a diagnosis the gate cannot use. It fails open, by design; froehlich
    63619743_29 and lucarelli 63968832_7 passed that way with relents `{}` and nothing said. The
    pass still goes through, but it is a relent, under its reason, and both replies tell the lead
    the check did not run."""
    mod = _agent
    best = _fit(version=15, bic=205.0)
    rows = [
        {"version": 15, "criterion_value": 205.0, "bic": 205.0},
        {"version": 20, "criterion_value": 260.0, "bic": 260.0},
    ]
    session = _RestoreSession(
        best,
        identifiability=_FLAT_16_ICS,
        versions=rows[:1],
        head=15,
        summaries={15: best},
        after_restore=after_restore,
    )
    fs = {}
    tool = mod._finish_tool(session, fs)

    refused = await _finish(tool)
    assert "FINISH REFUSED" in refused and "rank-deficient" in refused.lower()
    assert session.restored == []

    # The lead explores past v15 without beating it; its own last fit is identifiable.
    session.versions, session.head = rows, 20
    session.fit_summary = _fit(version=20, bic=260.0)
    session.identifiability = _IDENTIFIABLE

    note = prompts.FINISH_PARSIMONY_UNCHECKED_TMPL.format(why=why)
    armed = await _finish(tool)
    assert session.restored == [15]
    assert armed.startswith(prompts.FINISH_RESTORED_BEST_TMPL.format(best=15, head=20))
    assert "FINISH ARMED" in armed and "bic=205" in armed and armed.endswith(note)
    confirmed = await _finish(tool)
    assert confirmed.startswith("Recorded: finish") and confirmed.endswith(note)

    assert fs["confirmed"] is True
    assert fs["refusals"] == {"parsimony": 1}
    assert fs["relents"] == {f"parsimony_{reason}": 2}
    # the log names the server's own reason, so a run's .out says which source failed
    said = after_restore.get("stale_note") or after_restore["note"]
    assert f"[finish UNCHECKED] parsimony: diagnosis {reason} ({said})" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_parsimony_and_sigma_gates_compose():
    """The sigma gate runs FIRST: an implausible sigma is refused before the (expensive) FIM probe
    is even consulted, so a model with both problems is not double-charged in one call."""
    mod = _agent
    tool = _tool_for(
        mod,
        _fit(noise_sigma_inflated=True, noise_sigma_advice="fix the mechanism."),
        identifiability=_FLAT_16_ICS,
    )
    out = await _finish(tool)
    assert "FINISH REFUSED" in out and "implausible noise model" in out  # sigma wins this call


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
    session.identifiability = _FLAT_16_ICS  # refuses -> must disarm
    assert "FINISH REFUSED" in await _finish(tool)
    session.identifiability = _IDENTIFIABLE
    assert "FINISH ARMED" in await _finish(tool)  # re-arms, does NOT confirm
    assert (await _finish(tool)).startswith("Recorded: finish")


# A schwen-shaped geometry (gemot#134): `km_nExpID3` is a shipped detector constant that only the
# measurement table's observableParameters use. Its readout stays far below km's lower bound for any
# model, so the FIM is flat along it whatever the model is. The model's own rates carry an
# ill-conditioned but full-rank direction beside it. The server lists flat directions smallest
# eigenvalue first; these lists put the model's first, so the gate cannot lean on that order.
_KM = "km_nExpID3"
_KM_ONLY = {"eigenvalue": 0.0, "rel": 0.0, "participants": [{"id": _KM, "weight": 1.0}]}
_KM_AND_RATE = {
    "eigenvalue": 1e-12,
    "rel": 0.0,
    "participants": [{"id": _KM, "weight": 0.9}, {"id": "k3", "weight": 0.4}],
}
_RATES = {
    "eigenvalue": 2e-4,
    "rel": 4e-4,
    "participants": [{"id": "k1", "weight": 0.7}, {"id": "k2", "weight": -0.7}],
}
_SCHWEN_ROWS = [
    {"parameterId": _KM, "role": "observable_scaling"},
    {"parameterId": "sigma_insulin", "role": "noise_sigma"},
    *({"parameterId": k, "role": "kinetic"} for k in ("k1", "k2", "k3")),
]


def _schwen(
    *,
    rank,
    directions,
    parameters=_SCHWEN_ROWS,
    model_parameters=("k1", "k2", "k3"),
    n_at_bound=0,
):
    """A session whose FIM, over 4 estimated parameters inside their bounds, has ``rank`` and
    ``directions``. ``n_at_bound`` more are estimated but pinned, which the server counts in
    ``n_estimated`` and leaves out of the rank and the directions (PEtab-MCP#356)."""
    ident = {
        "available": True,
        "method": "fim+multistart",
        "fim": {
            "rank": rank,
            "n_estimated": 4 + n_at_bound,
            "n_at_bound": n_at_bound,
            "condition": 1e16,
            "rank_deficient": rank < 4,
        },
        "flat_directions": directions,
    }
    return _FakeSession(
        _fit(), identifiability=ident, parameters=parameters, model_parameters=model_parameters
    )


@pytest.mark.asyncio
async def test_a_deficiency_only_a_shipped_observation_parameter_carries_is_no_refusal():
    """schwen (gemot#134): every first refusal led with `[km_nExpID3·1.0]`, which no model edit can
    fix, and all 4 finished runs spent 45-100 min narrowing km's bounds to get past the gate. A
    rank deficiency lying only along such a parameter is not a refusal, and the model's merely
    ill-conditioned direction beside it does not become one (the rank-deficiency-only rule)."""
    fs = {}
    session = _schwen(rank=3, directions=[_RATES, _KM_ONLY])
    tool = _agent._finish_tool(session, fs)
    assert "FINISH ARMED" in await _finish(tool)
    assert (await _finish(tool)).startswith("Recorded: finish")
    assert fs["refusals"] == {} and fs["relents"] == {}


@pytest.mark.parametrize("n_at_bound", [1, 2])
@pytest.mark.asyncio
async def test_a_pinned_parameter_is_not_counted_as_a_deficient_direction(n_at_bound):
    """petab-mcp 0.1.0 takes the FIM's rank over the parameters inside their bounds while
    `n_estimated` still counts the pinned ones (PEtab-MCP#356). Read as `n_estimated - rank`, one
    pin beside schwen's km deficiency made the model's ill-conditioned but full-rank direction a
    deficient one, and two made more deficient directions than were listed: either way the gate
    refused a finish no edit can clear, the case gemot#134 took out."""
    fs = {}
    session = _schwen(rank=3, directions=[_RATES, _KM_ONLY], n_at_bound=n_at_bound)
    tool = _agent._finish_tool(session, fs)
    assert "FINISH ARMED" in await _finish(tool)
    assert fs["refusals"] == {}


@pytest.mark.asyncio
async def test_a_shipped_direction_is_never_named_beside_the_models_own():
    """A deficiency the model can reach still refuses, but names only what an edit reaches: the
    km-only direction is left out (and the refusal says something was), while a direction km
    shares with a model rate stays, since the rate is the model's to change."""
    session = _schwen(rank=2, directions=[_RATES, _KM_ONLY, _KM_AND_RATE])
    out = await _finish(_agent._finish_tool(session))
    assert out.startswith("FINISH REFUSED") and "cannot constrain" in out
    assert f"[{_KM}·1.0]" not in out
    assert f"[{_KM}·0.9, k3·0.4]" in out and "[k1·0.7, k2·-0.7]" in out
    assert prompts.FINISH_REFUSED_PARSIMONY_LEFT_OUT in out


@pytest.mark.asyncio
async def test_an_observation_parameter_the_model_declares_is_its_own():
    """A published model may reuse an observable's normaliser in its own math (`A_tot`, the
    server's warning-level reuse), and then an edit does reach it: such a parameter is the model's,
    and its direction refuses and is named as before."""
    session = _schwen(rank=3, directions=[_KM_ONLY], model_parameters=("k1", "k2", "k3", _KM))
    out = await _finish(_agent._finish_tool(session))
    assert out.startswith("FINISH REFUSED") and f"[{_KM}·1.0]" in out
    assert prompts.FINISH_REFUSED_PARSIMONY_LEFT_OUT not in out


@pytest.mark.parametrize(
    "parameters, model_parameters",
    [((), ("k1", "k2", "k3")), (_SCHWEN_ROWS, None), ("not rows", ("k1",))],
)
@pytest.mark.asyncio
async def test_without_readable_roles_the_parsimony_gate_refuses_as_before(
    parameters, model_parameters
):
    """Nothing is left out unless the reads show it out of reach: no observation roles, an
    unreadable model structure, or an odd `get_parameters` shape keep the refusal and its naming."""
    session = _schwen(
        rank=3, directions=[_KM_ONLY], parameters=parameters, model_parameters=model_parameters
    )
    out = await _finish(_agent._finish_tool(session))
    assert out.startswith("FINISH REFUSED") and f"[{_KM}·1.0]" in out


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

    assert await mod._model_committed(_probe_returning({})) is True  # key missing
    assert await mod._model_committed(_probe_returning(None)) is True  # no structured content


def test_no_model_nudge_text_is_actionable():
    mod = _agent
    assert "make_changes" in mod._NO_MODEL_NUDGE and "fit" in mod._NO_MODEL_NUDGE.lower()


# --- the invalid-comparison note on the ARM (petab_mcp.nesting) ----------------------------
# Written for blasi run 80ccf20efcbc020dc2becda41d14c62e, where the panel's ONLY "does adding
# structure help?" experiment (v3 -> v4) fit 3.984 WORSE with one more parameter, and the lead read
# the worse score as evidence the model was complete, then finished at 4 fitted ops on 5% of its
# budget. Since frohlich-lab/PEtab-MCP#276 the server flags only a child nested in its parent (that
# blasi edit replaced a factor in a rate law, so it now reads "not nested" and is not flagged), and
# keeps the flag until a valid additive experiment supersedes it; the gate must put it in front of
# the lead before it is allowed to confirm. It is a NOTE, not a refusal: the run may have other
# grounds to finish.
_NESTED_ADVISORY = (
    "INVALID COMPARISON vs v3: k rose 7->8 but neg_log_likelihood got WORSE by 3.98381, though the "
    "edit only added to v3 and its fitted values lie inside the bounds — a model nested in its "
    "parent cannot fit worse. Either the fit under-converged or a new term cannot be switched off "
    "inside its bounds. This does NOT show the added structure fails to help: re-fit with more "
    "starts before concluding anything from it."
)


@pytest.mark.asyncio
async def test_arm_surfaces_an_outstanding_invalid_comparison():
    mod = _agent
    tool = _tool_for(mod, _fit(invalid_comparison=_NESTED_ADVISORY))
    armed = await _finish(tool)
    assert "FINISH ARMED" in armed
    # the server's advisory is quoted verbatim, so its cause and its re-fit advice reach the lead
    assert _NESTED_ADVISORY in armed
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


# --- the gate judges the BEST version, the one the run is scored on (gemot#94) -------------------
# lucarelli runs were refused on a worse HEAD and spent fits rebuilding a clean model there ("The
# finish gate checks the current HEAD (v31)"; "I need HEAD on a clean model") while the version the
# judge would score sat in the store.
_ROWS = [
    {"version": 3, "criterion_value": 300.0, "bic": 300.0},
    {"version": 4, "criterion_value": 205.0, "bic": 205.0},
    {"version": 5, "criterion_value": 260.0, "bic": 260.0},
]


@pytest.mark.asyncio
async def test_the_gate_restores_and_judges_the_best_version_not_head():
    mod = _agent
    worse_head = _fit(version=5, bic=260.0, noise_sigma_inflated=True, noise_sigma_advice="x")
    best = _fit(version=4, bic=205.0, neg_log_likelihood=90.0)
    session = _FakeSession(worse_head, versions=_ROWS, head=5, summaries={4: best})
    tool = mod._finish_tool(session)
    armed = await _finish(tool)
    # HEAD's implausible sigma refuses nothing: the version judged is v4, and the lead is told so
    assert session.restored == [4]
    assert armed.startswith(prompts.FINISH_RESTORED_BEST_TMPL.format(best=4, head=5))
    assert "FINISH ARMED" in armed and "bic=205" in armed
    # ...and HEAD is now that version, so the confirming call restores nothing and confirms it
    confirmed = await _finish(tool)
    assert confirmed.startswith("Recorded: finish") and "bic=205" in confirmed
    assert session.restored == [4]


@pytest.mark.asyncio
async def test_the_gate_restores_nothing_when_head_is_the_best():
    mod = _agent
    session = _FakeSession(_fit(version=4), versions=_ROWS, head=4, summaries={4: _fit()})
    armed = await _finish(mod._finish_tool(session))
    assert "FINISH ARMED" in armed and "restored" not in armed
    assert "restore_version" not in session.calls


@pytest.mark.asyncio
async def test_a_fit_armed_fresh_confirms_once_the_gate_restores_it(capsys):
    """gemot#166, bruno 63612603_50/53. The server serves a restored version's summary with
    ``version: None`` (its stored simulation carries none). Keyed on that version, the fit the lead
    armed while it was HEAD (v4) re-armed when, after a worse v5, the gate restored it: the lead had
    sent its confirmation and was handed ARMED again. The gate names HEAD by describe_problem."""
    mod = _agent
    best = _fit(version=4, bic=205.0, neg_log_likelihood=90.0)
    session = _FakeSession(
        best, versions=_ROWS[:2], head=4, summaries={4: {**best, "version": None}}
    )
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    # The lead tries one more structure, which fits worse, and HEAD moves to it.
    session.versions, session.head, session.fit_summary = _ROWS, 5, _fit(version=5, bic=260.0)
    confirmed = await _finish(tool)
    assert session.restored == [4]
    assert "Recorded: finish" in confirmed and "bic=205" in confirmed
    # ...and the log names the version, where it printed `v=None`.
    assert "[finish CONFIRMED v=4]" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_log_names_a_restored_version_the_summary_leaves_unnamed(capsys):
    mod = _agent
    worse_head = _fit(version=5, bic=260.0)
    best = _fit(version=None, bic=205.0, neg_log_likelihood=90.0)
    session = _FakeSession(worse_head, versions=_ROWS, head=5, summaries={4: best})
    tool = mod._finish_tool(session)
    assert "FINISH ARMED" in await _finish(tool)
    assert "[finish ARMED v=4]" in capsys.readouterr().out
    assert "Recorded: finish" in await _finish(tool)


@pytest.mark.asyncio
async def test_a_version_the_constraints_now_break_is_never_the_one_judged():
    """The server says such a version cannot be restored and is not the best; ranking it first would
    leave the gate judging HEAD on a refused restore while a restorable best sits beside it."""
    mod = _agent
    rows = [{**_ROWS[1], "violates": ["c1"]}, _ROWS[0], _ROWS[2]]
    session = _FakeSession(_fit(version=3), versions=rows, head=3, summaries={5: _fit(version=5)})
    await _finish(mod._finish_tool(session))
    assert session.restored == [5]


@pytest.mark.asyncio
async def test_a_refused_restore_leaves_the_gate_judging_head():
    """Fail OPEN to the old behaviour: a restore the server refuses must not block or mislabel the
    finish."""
    mod = _agent
    session = _FakeSession(_fit(version=5, bic=260.0), versions=_ROWS, head=5, summaries={})
    armed = await _finish(mod._finish_tool(session))
    assert "FINISH ARMED" in armed and "bic=260" in armed and "restored" not in armed


_INFLATED = {"noise_sigma_inflated": True, "noise_sigma_collapsed": False}
_PLAUSIBLE = {"noise_sigma_inflated": False, "noise_sigma_collapsed": False}


@pytest.mark.asyncio
async def test_a_repaired_head_is_not_traded_back_for_a_gamed_best():
    """REGRESSION (review of #96): the best-scoring v4 has an inflated sigma, and the lead has just
    fitted v5, slightly worse but with a plausible one. Restoring v4 would refuse the finish on its
    sigma and undo the repair on every attempt, while the judge would score v5 anyway. The gate
    takes the judge's pick without its hidden reference: the best version the server's own sigma
    flags pass."""
    mod = _agent
    gamed = _fit(version=4, bic=205.0, noise_sigma_inflated=True, noise_sigma_advice="x")
    session = _FakeSession(
        _fit(version=5, bic=260.0),
        versions=_ROWS,
        head=5,
        summaries={4: gamed},
        metas={4: _INFLATED, 5: _PLAUSIBLE},
    )
    armed = await _finish(mod._finish_tool(session))
    assert session.restored == []
    assert "FINISH ARMED" in armed and "bic=260" in armed and "restored" not in armed


@pytest.mark.asyncio
async def test_a_gamed_best_is_passed_over_for_the_next_plausible_version():
    mod = _agent
    rows = [*_ROWS, {"version": 6, "criterion_value": 400.0, "bic": 400.0}]
    session = _FakeSession(
        _fit(version=6, bic=400.0),
        versions=rows,
        head=6,
        summaries={5: _fit(version=5, bic=260.0)},
        metas={4: _INFLATED, 5: _PLAUSIBLE},
    )
    armed = await _finish(mod._finish_tool(session))
    assert session.restored == [5]
    assert armed.startswith(prompts.FINISH_RESTORED_BEST_TMPL.format(best=5, head=6))


@pytest.mark.asyncio
async def test_when_every_version_is_gamed_the_best_scoring_one_is_judged():
    """As the judge does: with no plausible version to prefer, the run is reported on its own best
    score, and the gate's sigma refusal speaks to it."""
    mod = _agent
    gamed = _fit(version=4, bic=205.0, noise_sigma_inflated=True, noise_sigma_advice="x")
    session = _FakeSession(
        _fit(version=5, bic=260.0, noise_sigma_inflated=True, noise_sigma_advice="x"),
        versions=_ROWS,
        head=5,
        summaries={4: gamed},
        metas={3: _INFLATED, 4: _INFLATED, 5: _INFLATED},
    )
    out = await _finish(mod._finish_tool(session))
    assert session.restored == [4]
    assert "FINISH REFUSED" in out and "implausible noise model" in out


# The CHI2 gate: a run judged by chi2 whose fit estimated no noise sigma. `_GIVEN` is the
# noise_sigma_report of such a fit (empty: nothing estimated); `_STORE` one fitted version, HEAD and
# best, so the gate restores nothing and the tests can add versions to mark new work.
_GIVEN: dict = {}
_STORE = [{"version": 5, "criterion_value": 210.0, "bic": 210.0}]


def _chi2_fit(p, report=_GIVEN):
    return _fit(chi2_gof_pvalue=p, noise_sigma_report=report)


def _chi2_gate(fit_summary, *, adequacy="chi2", finish_state=None):
    session = _FakeSession(fit_summary, versions=[dict(r) for r in _STORE], head=5)
    tool = _agent._finish_tool(session, finish_state, adequacy=adequacy)
    return session, tool


def _new_version(session):
    """A new fitted version, worse than the best, so the best (and the fit judged) stays put."""
    v = max(r["version"] for r in session.versions) + 1
    session.versions.append({"version": v, "criterion_value": 300.0, "bic": 300.0})


@pytest.mark.asyncio
async def test_chi2_gate_refuses_a_failing_fit_on_given_noise():
    """The dessauges case (gemot_iclr_2026 sweep 63372750): judged by chi2 against given SEMs, 4 of
    5 leads finished on fits the judge then scored inadequate. A failing chi2 there is refused,
    and the refusal is counted for the run's result."""
    fs = {}
    _, tool = _chi2_gate(_chi2_fit(1e-6), finish_state=fs)
    out = await _finish(tool)
    assert out.startswith("FINISH REFUSED") and "p=1e-06" in out
    assert fs["refusals"] == {"adequacy": 1} and fs.get("confirmed") is not True


@pytest.mark.asyncio
async def test_chi2_gate_never_reads_chi2_against_an_estimated_sigma():
    """Issue #54: with a sigma estimated, chi2 pins near n and refusing on it chased an
    unreachable target. A non-empty noise_sigma_report, or an unreadable one, never refuses."""
    for report in ({"noise_obs": {"fitted": 1.0}}, None):
        _, tool = _chi2_gate(_chi2_fit(1e-6, report=report))
        assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_chi2_gate_is_off_unless_the_run_is_judged_by_chi2():
    """A run judged against a reference (no `adequacy`) or by another test is never refused on
    chi2, however small its p."""
    for adequacy in (None, "residual_structure", "replicate_lof"):
        _, tool = _chi2_gate(_chi2_fit(1e-6), adequacy=adequacy)
        assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_chi2_gate_passes_an_adequate_fit():
    _, tool = _chi2_gate(_chi2_fit(0.2))
    assert "FINISH ARMED" in await _finish(tool)


@pytest.mark.asyncio
async def test_chi2_gate_cannot_be_worn_down_by_finishing_again():
    """A refusal is SPENT only by a finish that follows a new version: re-calling finish on
    unchanged work (the pattern of a lead calling finish after every refusal until the gate
    relented) is refused every time and brings the relent no closer. After
    _MAX_ADEQUACY_FINISH_REFUSALS spent refusals, each after new work, the finish goes through."""
    n = _agent._MAX_ADEQUACY_FINISH_REFUSALS
    fs = {}
    session, tool = _chi2_gate(_chi2_fit(1e-6), finish_state=fs)
    for spent in range(1, n + 1):
        out = await _finish(tool)  # the first, or after new work: refused, and spent
        assert out.startswith("FINISH REFUSED")
        if spent < n:
            for _ in range(3):  # no new version since: refused again, and NOT spent
                assert (await _finish(tool)).startswith("FINISH REFUSED")
            _new_version(session)
    assert "FINISH ARMED" in await _finish(tool)  # every refusal spent: the finish goes through
    assert fs["relents"] == {"adequacy": 1}
    assert fs["refusals"] == {"adequacy": n + 3 * (n - 1)}


@pytest.mark.parametrize(
    "report, p",
    [({}, 0.01), ({}, 0.05), ({}, 0.2), ({"noise_obs": {}}, 0.01), (None, 0.01), ({}, None)],
)
def test_the_gate_reads_chi2_as_the_judge_does(report, p):
    """The gate refuses exactly the fits the judge scores inadequate: it applies
    scenario.chi2_passes because the agent package cannot import the judge (it pulls in
    petab_mcp, see test_import_lightness), so the two rules are locked together here."""
    from gemot.runner.scoring import _reference_free_adequacy
    from gemot.scenario import chi2_passes

    assert chi2_passes(report, p) == _reference_free_adequacy("chi2", report, p, None)["passed"]


@pytest.mark.asyncio
async def test_an_inadequate_over_parameterised_model_hears_adequacy_first():
    """gemot#134: the parsimony refusal used to return before the chi2 check was reached, so an
    inadequate, rank-deficient model was told only that some parameters were unconstrained.
    dessauges 90 (p 1e-23) then spent its last 11 fits on equivalent rewrites until the streak cap
    ended it. The chi2 refusal comes first, without the FIM probe; once the model passes chi2, the
    parsimony refusal follows."""
    fs = {}
    session = _FakeSession(
        _chi2_fit(1e-23),
        identifiability=_FLAT_16_ICS,
        versions=[dict(r) for r in _STORE],
        head=5,
    )
    tool = _agent._finish_tool(session, fs, adequacy="chi2")
    out = await _finish(tool)
    assert out.startswith("FINISH REFUSED") and "fails chi2" in out
    assert "cannot constrain" not in out and "get_fit_diagnosis" not in session.calls
    session.fit_summary = _chi2_fit(0.2)
    out = await _finish(tool)
    assert out.startswith("FINISH REFUSED") and "cannot constrain" in out
    assert fs["refusals"] == {"adequacy": 1, "parsimony": 1}
