"""The independent verdict: score the built model against the true model by the run's criterion.

Deliberately structure-agnostic — it does not check whether the agent found "the" mechanism, only
whether what it built explains the data at a competitive parameter count (``delta_criterion <=
scenario.bic_threshold``), with the noise-sigma plausibility gate as the second condition so a
good score bought with a gamed noise model does not pass. The scenario's own ``diagnostics`` run
report-only alongside.

The gate uses the complexity criterion the SERVER ranked models by (``bic`` unless configured —
:mod:`petab_mcp.criteria`), so a run that searched by AICc is graded by AICc. Canonical BIC is
recorded on every run regardless (``bic`` / ``delta_bic``), which keeps arms comparable on one
fixed scale; under the default criterion the two are the same number.

Scores the run's BEST checkpoint, not whatever HEAD happens to be: a run that ended on a worse
model is rolled back first.

A ``reference_free`` scenario (a model-free data package) has no true model, so there is no Δ to
gate on. Its bar is instead the absolute goodness-of-fit test the scenario declares
(:func:`scoring._reference_free_adequacy`), beside the same valid-fit and sigma-plausibility
conditions; every reference-based number is recorded as ``None`` and ``judge_mode`` says which bar
decided ``success``.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import time
from pathlib import Path

from .. import ablations
from ..agent.circuit_breaker import _is_call_timeout
from .clock import HELD_OUT_TOOL, _ToolClock
from ..paths import validation_measurements_path
from .provenance import _bic_reference, _judge_mode, _now_iso, _reference_score
from ..scenario import Baseline, JudgeContext, Scenario
from .scoring import (
    _noise_sigma_flag,
    _reference_free_adequacy,
    _reference_sigma_flag,
    _residual_structure_fields,
    _validation_fields,
)


async def _head_version(call) -> "int | None":
    """The current HEAD version number, read off ``describe_problem``.

    There is no separate ``current_version`` tool — ``describe_problem`` carries it. ``None`` when
    the read fails, so a caller can fall back rather than abort. ``-1`` means nothing has been
    fit: versions only exist for scored models.
    """
    try:
        desc = await call("describe_problem")
    except Exception:  # pragma: no cover - defensive; callers degrade
        return None
    return int(desc.get("current_version", -1)) if isinstance(desc, dict) else None


#: How many fitted versions the scoring scan will inspect, best-scoring first — EVERY one of them.
#: Each costs a restore and a score read (no fit, no compile), and the count is already bounded by
#: the run's own turn budget: a version exists only where a `make_changes` call fitted one, so the
#: recursion limit caps the list at a few hundred in the worst case and at 10-30 in practice. A
#: tighter cap only buys a few seconds, and buys them by giving up the run's best scoreable model —
#: exactly the failure this scan exists to prevent, reintroduced one rung further down.
_SCORING_SCAN_LIMIT = None


async def _restore_best_checkpoint(call, reference=None) -> None:
    """Make the best model the judge can legitimately score the current HEAD before scoring.

    Agents wander off their best model and finish worse (verified pass→fail flips on a credit
    abort, e.g. armistead v56 → v66), so the judge must score the BEST fitted version, not the
    last-committed one. Runs on ANY termination that reaches the judge (early-stop, wall-time,
    recursion limit, clean finish). Best-effort throughout — any probe failure leaves HEAD in place.

    "Best" is NOT simply the lowest score. A model whose estimated sigma is implausible against the
    reference's own fitted sigmas fails :func:`_judge`'s integrity gate, so scoring it throws the run
    away while a clean, nearly-as-good version sits unscored in the same version store. The scan
    therefore walks EVERY fitted version best-first under the RUN'S OWN criterion (see
    :func:`_scoreable_versions`; identical to best-BIC under the default) and stops at the first one
    that would clear the gate, falling back to the plain best-scoring model when NONE is clean (an
    all-gamed run is still reported on its own best model, exactly as before, and still fails).

    Why the candidate set is every fitted version and not the server's ``best_version`` checkpoint
    (brannmark, run 9285a352): ``v5`` scored bic 344.175 but improved on ``v4`` by only 0.70 — under
    ``NEW_BEST_MIN_DELTA_SCORE`` (2.0) — so the checkpoint stayed at ``v4`` (bic 344.875) and the judge
    scored the worse model. That run failed on the sigma gate either way (all six of its versions
    fitted sigmaYDosR at 0.24-0.28 of the reference's, well under the 1/3 floor), so the hysteresis
    cost it nothing there; it is a pass waiting to be discarded the first time the two versions
    differ in plausibility. The threshold is right for the streak and plateau logic, which need a
    stable notion of progress, and wrong for a final decision, which should see every version.

    Nor can the server make this choice: the plausibility question here is asked against the TRUE
    model's fitted sigmas, which live in the scenario's reference and are deliberately not visible
    to the session (see :func:`scoring._reference_sigma_flag`). Only the judge holds both halves.

    No fitted version at all means no ``make_changes`` call ever produced one — the agent built
    nothing, or every fit failed. There is nothing to score either way, so the judge scores HEAD and
    the run is reported on its own (empty or unfit) merits.
    """
    candidates = await _scoreable_versions(call)
    if not candidates:  # nothing fitted, or an older server without list_versions
        return await _restore_checkpoint_only(call)
    head = await _head_version(call)
    fallback = candidates[0]
    chosen, clean = fallback, False
    for version in candidates[:_SCORING_SCAN_LIMIT]:
        flagged = await _sigma_flagged(call, version, reference)
        # `None` is unassessable — no stored report, or one the gate cannot read. The gate will not
        # fail that version either, so it is scoreable and the scan stops on it exactly as it does
        # on a clean one. (A whole store of them, e.g. a version dir written before the report was
        # persisted, therefore lands on the best-scoring version: the pre-scan behaviour.)
        if flagged is not True:
            chosen, clean = version, True
            break
    # ONE restore, of the version actually chosen. The scan itself is read-only: restoring each
    # candidate to look at it would mutate the working state N times and write N rollback entries
    # into the provenance the run is judged on, to read numbers already stored beside each version.
    #
    # ANNOUNCED ONLY ONCE IT SUCCEEDS. A failed restore leaves HEAD where the AGENT left it — which,
    # now that the scan no longer walks the versions, can be any model at all, not the runner-up —
    # and printing the choice first would put a version number in the log that nothing scored.
    if not await _restore(call, chosen, head):
        print(f"[judge] could not restore v{chosen}; scoring HEAD v{head} instead")
        return
    if not clean:
        # Every candidate would fail the sigma gate. Score the plain best-scoring one: the run's
        # own lowest-scoring model is the honest thing to report, and the gate still fails it.
        print(
            f"[judge] none of the run's {len(candidates)} fitted versions has a plausible noise "
            f"sigma — scoring best-scoring v{fallback}"
        )
    elif chosen != fallback:
        print(
            f"[judge] scoring v{chosen} — the best-scoring version whose noise sigma is "
            f"plausible (v{fallback} scores better but its sigma would fail the gate)"
        )


def _implausible_sigma_text(flag: dict) -> str:
    """Which way an implausible noise sigma failed, for the verdict line."""
    return " and ".join(
        w
        for w, on in (
            ("inflated (absorbing misfit)", flag["inflated"]),
            ("collapsed (overfitting the noise)", flag["collapsed"]),
        )
        if on
    )


def _candidate_k(params, final, estimated=None) -> "tuple[int, int, list[str]]":
    """``(k, k_estimated, hardcoded_constants)`` for the model being scored.

    ``k_estimated`` counts ``get_parameters`` rows with ``estimate`` truthy — NOT
    ``len(fitted_parameters)``, which also counts fixed params (noise σ, the reconcile's parametric
    initial conditions) and would inflate the penalty and diverge from how reference.json counts k.
    Rows the read appends for parameters the PROBLEM ships but the model never declared
    (``in_model: false``, since #131) are not free parameters of the model being scored and are
    skipped — counting them charged a candidate that left a shipped estimated row undeclared for
    a parameter it never had (brannmark ships eight such rows), and made this k disagree with the
    server's ``n_estimated`` by exactly those rows. ``hardcoded_constants`` is the list
    ``get_fit_summary`` reports (petab_mcp.literals): the
    numeric literals in the model's math outside the free structural set, each of which the
    server's own bic prices as one parameter. The judge adds them too, else the headline ΔBIC would
    reward replacing the parameter vector with numerals (issue #180). Re-derived here from the two
    reads rather than copied off the server's ``k``, so the two stay independently derived yet
    equal. A summary that lacks the list (a server predating it) charges none.

    ``estimated`` replaces the rows for the null arm: the ids the server's reconcile estimates for
    its delivery (:func:`.barebones._reconciled`, issue #1); the server that scores the delivery
    serves it as shipped, unreconciled. The panel passes none: its server reconciles every model
    before it is fitted.
    """
    k_estimated = (
        sum(
            1
            for p in params
            if isinstance(p, dict) and p.get("estimate") and p.get("in_model", True)
        )
        if isinstance(params, list)
        else 0
    )
    hardcoded = final.get("hardcoded_constants") if isinstance(final, dict) else None
    hardcoded = [str(c) for c in hardcoded] if isinstance(hardcoded, list) else []
    if estimated is not None:
        k_estimated = len(estimated)
    return k_estimated + len(hardcoded), k_estimated, hardcoded


def _fitted_parameters(params) -> "dict[str, float] | None":
    """The scored version's estimated parameters by id, at their FITTED values.

    A fit writes its result into the parameter table's ``nominalValue`` (see
    ``fitting._slim_fitted_parameters``), so the estimated rows carry the vector and no extra read
    is needed. Exactly the rows :func:`_candidate_k` counts: an ``in_model: false`` row is a
    parameter the PROBLEM ships that this model never declared, so it is not a parameter of the
    model being scored and does not belong in its vector.

    ``None`` rather than ``{}`` where the read is unusable, so "no vector recorded" stays
    distinguishable from "a model with nothing estimated".
    """
    if not isinstance(params, list):
        return None
    out: dict[str, float] = {}
    for parameter in params:
        if not (
            isinstance(parameter, dict)
            and parameter.get("estimate")
            and parameter.get("in_model", True)
        ):
            continue
        value = parameter.get("nominalValue")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(parameter.get("parameterId"))] = float(value)
    return out


def _stored_k_and_n(stored) -> "tuple[int | None, int | None]":
    """``(k, n)`` for a stored version: its ESTIMATED-parameter count and its data-point count.

    The band the sigma gate applies depends on how much freedom the model spent (see
    :func:`scoring._reference_sigma_flag`), and for a version the judge has not restored those come
    off the stored payload rather than off live ``get_parameters``/``get_fit_summary``. ``k`` counts
    rows with ``estimate`` truthy, as the judge does — but off THAT VERSION's stored parameter table
    rather than the live one, and the two need not agree to the row: ``get_parameters`` also reports
    parameters the PROBLEM ships that the model never declared (``in_model: false``), and brannmark
    ships eight all-estimated rows a candidate may leave undeclared. The consequence is bounded: k
    enters only as the scale factor ``1 - k/n`` on a band's degrees of freedom, so a handful of rows
    shifts a band slightly and cannot move a sigma across it the way a wrong k would move a score.
    ``(None, None)`` when either is unreadable: the band then has no df to work from and applies the
    fixed tolerance factor, so the sigma is still judged, just without df-awareness.
    """
    if not isinstance(stored, dict):
        return None, None
    params = stored.get("parameters")
    k = (
        sum(1 for p in params if isinstance(p, dict) and p.get("estimate"))
        if isinstance(params, list)
        else None
    )
    simulation = stored.get("simulation")
    n = simulation.get("n_data_points") if isinstance(simulation, dict) else None
    return k, (n if isinstance(n, int) else None)


async def _scoreable_versions(call) -> "list[int]":
    """Every FITTED version, best first UNDER THE RUN'S OWN CRITERION — the candidate set for
    the scoring scan.

    Read off ``list_versions`` rather than ``get_progress`` so a version the checkpoint's
    minimum-improvement threshold skipped is still a candidate (see :func:`_restore_best_checkpoint`).
    A version with no finite score was never scored and cannot be scored now, so it is dropped.

    Ordered by ``criterion_value`` (identical to ``bic`` under the default BIC), NOT by canonical
    bic: this scan complements the session's own best-score checkpoint, so its notion of "best
    first" has to be the same one the run used to build that checkpoint — otherwise, under a
    non-BIC criterion, the scan would start from a model the run itself never preferred. The same
    criterion then GRADES the chosen version in :func:`_judge`, with canonical BIC recorded
    alongside (see :mod:`petab_mcp.criteria`). ``bic`` is the fallback when ANY version predates
    the recorded criterion, applied to the WHOLE set rather than per-version so one ordering never
    mixes two scales.
    """
    try:
        versions = await call("list_versions")
    except Exception:  # pragma: no cover - older server: caller falls back to the checkpoint
        return []
    rows = [v for v in versions if isinstance(v, dict)] if isinstance(versions, list) else []
    # Rows that could be ranked AT ALL. One carrying neither score is dropped by the loop below
    # either way, so it must not get a vote on which key the rest are sorted by.
    rows = [v for v in rows if "criterion_value" in v or "bic" in v]
    # ONE scale for the WHOLE ordering: `criterion_value` only when every candidate carries it,
    # else canonical `bic` for all of them. Falling back per-version instead would sort a store
    # that spans this upgrade on two scales at once — an early version's BIC against a later
    # one's AICc — leaving the candidate order arbitrary between the two groups. A row whose
    # `criterion_value` is present but NULL still counts as carrying it: that fit is unscoreable
    # under this criterion and is meant to be dropped, not to trigger a fallback.
    key = "criterion_value" if all("criterion_value" in v for v in rows) else "bic"
    scored = []
    for v in rows:
        score, num = v.get(key), v.get("version")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and isinstance(num, int)
        ):
            scored.append((float(score), num))
    scored.sort()
    return [num for _, num in scored]


async def _sigma_flagged(call, version: int, reference) -> "bool | None":
    """Would ``version``'s noise model fail :func:`_judge`'s integrity gate? ``None`` when
    unassessable.

    The SAME verdict the gate itself computes — ``_reference_sigma_flag`` against the reference's
    fitted sigmas, falling back to the data-std band — read off the report the FIT stored beside
    that version (:func:`petab_mcp.fitting.build_sim_payload_for_version`). A read, not a rollback:
    the scan can ask about every version without touching the working state.

    Anything unreadable is ``None``: an unassessable version must not be silently skipped in favour
    of a worse one, since the gate will not fail it either. A version stored before the report was
    persisted reads that way too, which is what makes the scan degrade to plain best-scoring on an
    old store rather than mis-rank it.
    """
    try:
        stored = await call("get_version", version=version)
    except Exception:  # pragma: no cover - unreadable version: no verdict
        return None
    simulation = stored.get("simulation") if isinstance(stored, dict) else None
    report = simulation.get("noise_sigma_report") if isinstance(simulation, dict) else None
    if not report:
        return None
    k_v, n_v = _stored_k_and_n(stored)
    flag = _reference_sigma_flag(report, reference, k=k_v, n=n_v) or _noise_sigma_flag(report)
    if not flag:
        return None
    return bool(flag["inflated"] or flag["collapsed"])


async def _restore(call, version: int, head: "int | None") -> bool:
    """Put ``version`` in the working state, or say it could not be done."""
    if version == head:
        return True
    try:
        await call(
            "restore_version",
            version=version,
            justification="scoring scan: evaluate this version before the judge scores",
        )
        return True
    except Exception as exc:  # pragma: no cover - skip a version we cannot restore
        print(f"[judge] could not restore v{version} ({type(exc).__name__}: {exc})")
        return False


async def _restore_checkpoint_only(call) -> None:
    """The pre-scan behaviour, for a server that cannot list versions: restore ``best_version``."""
    try:
        progress = await call("get_progress")
    except Exception:  # pragma: no cover - older server without the tool: score HEAD
        return
    best = progress.get("best_version") if isinstance(progress, dict) else None
    if best is None:
        return
    head = await _head_version(call)
    if best == head:
        return  # already on the best model
    if await _restore(call, best, head):
        print(f"[judge] restored best-scoring checkpoint v{best} (was HEAD v{head}) before scoring")


async def _judge(
    scenario: Scenario,
    call,
    baseline: Baseline,
    work: Path,
    meta: dict,
    clock: "_ToolClock",
    t0: float,
    estimated: "list[str] | None" = None,
) -> "tuple[int, dict]":
    """Judge the repaired model against the true model by the run's criterion (structure-agnostic).

    Success ⇔ the repaired model can't be rejected relative to the true (data-generating)
    model — ``score(candidate) - score(true) <= scenario.bic_threshold``, both scored under the
    complexity criterion the SERVER ranked models by (``bic`` unless configured — see
    :mod:`petab_mcp.criteria` and ``docs/RATIONALE.md``) — AND its estimated noise
    sigma is plausible (not inflated to absorb misfit or collapsed to overfit the noise; see
    ``_noise_sigma_flag``). The sigma check is a HIDDEN integrity gate: it never reaches the agent,
    so a good score bought with an implausible noise model can't slip through and can't be gamed
    around. We don't require any particular rate-law FORM (models need not be structurally
    identifiable) and we don't special-case invented species — the criterion's complexity penalty
    (BIC's ``k·ln(n)``) makes an extra parameter earn its keep. A χ² goodness-of-fit p-value
    (residuals vs the known noise) is reported alongside. The scenario's optional ``diagnostics``
    hook adds report-only detail.

    For a ``reference_free`` scenario the criterion gate is replaced, not relaxed: success ⇔ a valid
    fit AND a plausible sigma (the data-std band, there being no reference sigma) AND the scenario's
    declared adequacy test PASSES — one it could not assess is not a pass.

    A candidate scored on a different number of data points than the reference has no Δ against
    it: the deltas are withheld (``n_mismatch``) and the run fails. ``estimated`` is the null's (see
    :func:`_candidate_k`).

    Returns ``(exit_code, result)``: the process exit code (0 on success, 1 otherwise) and
    the machine-readable result dict (also written to ``work/result.json``), so the caller
    can stamp the outcome onto the run span.
    """
    from petab_mcp.criteria import CANONICAL, Criterion

    from ..scenario import chi2_gof_pvalue

    # Restore + score the best model this judge can legitimately score — not whatever was committed
    # last, and not a lower-scoring version whose noise model the gate below would reject while a
    # clean one sits unscored (see the helper). Handed the reference so the scan asks the SAME sigma
    # question the gate does.
    await _restore_best_checkpoint(call, scenario.reference())

    structure = await call("get_model_structure")
    reactions = structure.get("reactions", []) or []
    current_species = set(structure.get("species", []) or [])
    new_species = sorted(current_species - baseline.species)

    final = await call("get_fit_summary")
    # A degenerate or never-built model (e.g. the agent authored nothing, or a fit that produced
    # a nan/inf objective) has no usable neg_log_likelihood — get_fit_summary returns None, or a
    # non-finite value. Treat any of those as "no valid fit" (final_nll = None) and score the run
    # as a clean failure below, rather than crashing the judge on float(None) / a non-finite score.
    raw_nll = final.get("neg_log_likelihood") if isinstance(final, dict) else None
    try:
        final_nll = float(raw_nll)
    except (TypeError, ValueError):
        final_nll = None
    if final_nll is not None and not math.isfinite(final_nll):
        final_nll = None
    n = int(final.get("n_data_points") or 0) if isinstance(final, dict) else 0
    # k = number of ESTIMATED parameters (estimate=1), NOT len(fitted_parameters): the latter
    # also counts fixed params (noise σ, and the reconcile-added parametric initial
    # conditions), which would inflate the penalty and diverge from how reference.json counts
    # k. Both sides must count free parameters the same way. PLUS the candidate's hardcoded
    # constants (petab_mcp.literals): a fitted value frozen into a rate law is priced as the
    # parameter it is, exactly as the server's own bic prices it — otherwise the headline ΔBIC
    # would reward replacing the parameter vector with numerals (issue #180). The reference has
    # none by construction: its published constants are fixed rows or the author's literals. For
    # a null delivery the rows are those the server's reconcile estimates for it (#1).
    params = await call("get_parameters")
    k, k_estimated, hardcoded = _candidate_k(params, final, estimated)
    chi2 = final.get("chi2") if isinstance(final, dict) else None
    chi2 = float(chi2) if chi2 is not None else None
    # Residual-structure tests (sign, runs, replicate lack-of-fit) of the SCORED model — the checkpoint was already
    # restored above, so this is the fit that gets judged. Recorded for post-hoc analysis: e.g.
    # whether runs that passed the criterion gate also had structure-free residuals, and whether the
    # tests were underpowered. Report-only: a diagnosis failure must never fail the run (which
    # would leave no result.json at all — the exact unscored-run outcome the harness avoids), so
    # the call is guarded and degrades. The rolled-up verdict comes from `final` (get_fit_summary,
    # which carries it) and the per-observable rows from the diagnosis — so this asks for the RESIDUAL
    # section only: nothing here reads the parameter geometry, and the projection below discards it.
    # The ROWS call is guarded on its own: the verdict half is already in hand when it runs, and
    # wrapping both in one try threw that away over a failure to read the detail behind it.
    rollup = final.get("residual_structure") if isinstance(final, dict) else None
    try:
        # max_observables=0: NO cap. The agent-facing default (12) exists for its context
        # window; this projection lands in result.json permanently and is what every later
        # analysis of the run reads, so it must carry every observable.
        diagnosis = await call("get_fit_diagnosis", sections="residuals", max_observables=0)
    except Exception as exc:  # noqa: BLE001 - report-only diagnostic; never fail the judge on it
        print(f"  (residual-structure detail unavailable: {type(exc).__name__}: {exc})")
        diagnosis = None
    try:
        residual_structure = _residual_structure_fields(diagnosis, rollup)
    except Exception as exc:  # noqa: BLE001 - report-only diagnostic; never fail the judge on it
        print(f"  (residual-structure diagnosis unavailable: {type(exc).__name__}: {exc})")
        residual_structure = None
    # Keep None as None (an unreadable probe), distinct from -1 (nothing fit). A diff needs two
    # versions, so it is skipped unless HEAD is at least v1.
    head = await _head_version(call)
    diff = (
        await call("diff_versions", from_version=0, to_version=head)
        if head is not None and head > 0
        else {}
    )
    # The effective parameter count of the SCORED fit (identifiability.effective_parameter_count):
    # k_eff well below k says the data did not determine that many parameters. Read off the STORED
    # version, like the sigma reports in the scan above — it is on no result the agent sees, and
    # deriving it needs the built objective. Report-only: a store without it records None.
    k_eff = None
    try:
        stored = await call("get_version", version=head) if head is not None else None
        sim = stored.get("simulation") if isinstance(stored, dict) else None
        k_eff = sim.get("k_eff") if isinstance(sim, dict) else None
    except Exception:  # noqa: BLE001 - report-only
        k_eff = None
    errors = await call("get_errors", limit=20)
    # baseline.nll is None for a zero-shot build (no model shipped → nothing to score).
    delta = (
        (baseline.nll - final_nll) if (baseline.nll is not None and final_nll is not None) else None
    )

    # Estimated-noise-sigma plausibility of the SCORED model. A wrong model can reach a good score
    # by making the noise model implausible — inflating sigma to absorb misfit, or collapsing it to
    # overfit the noise — so an implausible sigma disqualifies an otherwise-passing score. This is a
    # HIDDEN integrity gate: applied post-run in the judge and NEVER surfaced to the agent (which
    # only ever sees its own noise_sigma_report, not this verdict), so it can't be gamed around.
    # The raw report is stored in result.json too (it wasn't before) to audit gaming after the fact.
    noise_sigma_report = final.get("noise_sigma_report") if isinstance(final, dict) else None
    ref = scenario.reference()
    judge_mode = _judge_mode(scenario)
    reference_free = judge_mode == "reference_free"

    # HELD-OUT (validation) score of the SCORED model — its likelihood on measurements it was never
    # fitted to, against the same number for the true model (reference.json's `validation`). The one
    # thing ΔBIC structurally cannot see: a mechanism that is wrong only under conditions the
    # training data does not cover scores identically to the right one on the training data (see
    # petab_mcp.validation_data). REPORT-ONLY — it does not enter `success` below.
    #
    # Asked ONLY of an example that actually ships held-out data. The predicate is the file, not the
    # reference's `validation` block: an example whose data is materialized but whose reference has
    # not been refit yet then reports a real score with `reference_nll: null`, which is the visible
    # form of that misconfiguration — gating on the block instead would hide it. The other
    # seventeen examples have no such file, so they skip the call entirely rather than printing an
    # "unavailable" line on every run for a check they were never meant to have.
    #
    # Guarded and best-effort, like the residual diagnosis above: a failure to read it must never
    # cost the run its result.json. `_validation_fields` turns "asked but not answered" into an
    # explicit status rather than a blank, so a sweep can tell it from a clean skip — and a call
    # that ran out of time from one that failed, since the tool was there all along (gemot#57).
    # How long the call may take is the clock's to decide (`clock._wrap_call_timeout`).
    validation = None
    if validation_measurements_path(scenario.example_dir).is_file():
        waited_s = None
        asked_at = time.monotonic()
        try:
            validation_payload = await call(HELD_OUT_TOOL)
        except Exception as exc:  # noqa: BLE001 - report-only; never fail the judge on it
            validation_payload = None
            if _is_call_timeout(exc):
                waited_s = time.monotonic() - asked_at
            print(
                f"  (held-out validation score "
                f"{'unavailable' if waited_s is None else 'timed out'}: {type(exc).__name__}: {exc})"
            )
        try:
            validation = _validation_fields(validation_payload, ref, asked=True, waited_s=waited_s)
        except Exception as exc:  # noqa: BLE001 - report-only diagnostic
            print(f"  (held-out validation projection unavailable: {type(exc).__name__}: {exc})")

    # The causal_constraints arm's record (gemot.causal_constraints): the refusals per constraint,
    # the first version making them all, and the judged model's behaviour on each indirect one.
    # Report-only and guarded like the held-out read above; its simulation compiles, so it runs in
    # a thread. The entity map is the arm's own, the empty one indra_statements starts the server
    # with (`runner.main._empty_entity_map`), so the rebuild reads the rows' agents with none.
    causal_constraints = None
    if ablations.enabled("causal_constraints"):
        from .. import causal_constraints as constraints_record

        try:
            with tempfile.TemporaryDirectory(prefix="gemot_constraints_") as build:
                causal_constraints = await asyncio.to_thread(
                    constraints_record.record,
                    work,
                    scenario.constraints_file,
                    clock,
                    head,
                    entity_map=None,
                    build_dir=build,
                )
        except Exception as exc:  # noqa: BLE001 - report-only; never fail the judge on it
            print(f"  (causal-constraints record unavailable: {type(exc).__name__}: {exc})")
            causal_constraints = {"error": f"{type(exc).__name__}: {exc}"}

    # PREFER the reference-sigma comparison (#118): a candidate sigma is judged against the TRUE
    # model's own fitted sigma, not against a fraction of the marginal data std (which conflates
    # explained and unexplained variance, so no fixed band holds the truth). Falls back to the
    # data-std band for a reference.json written before `noise_sigma` was recorded — regenerate
    # with `python -m petab_agent_bench --refit <example>` to switch an example over.
    reference_sigma_flag = _reference_sigma_flag(noise_sigma_report, ref, k=k, n=n)
    noise_sigma_flag = reference_sigma_flag or _noise_sigma_flag(noise_sigma_report)
    sigma_basis = "reference" if reference_sigma_flag else "data_std"
    sigma_implausible = bool(
        noise_sigma_flag and (noise_sigma_flag["inflated"] or noise_sigma_flag["collapsed"])
    )

    # Canonical BIC of candidate and reference, RECORDED for every run whatever the criterion
    # (docs/RATIONALE.md) even though the gate below follows the criterion.
    bic_cand = CANONICAL["bic"].value(final_nll, k, n)
    bic_ref = _bic_reference(ref)
    delta_bic = (bic_cand - bic_ref) if (bic_cand is not None and bic_ref is not None) else None

    # EVERY criterion, candidate against reference, so a run can be read under any of them
    # afterwards. `None` wherever a side has no value: aicc at n <= k+1, tic without a k_eff —
    # which the reference only carries once regenerated with `python -m petab_agent_bench --refit`.
    criteria_scores = {}
    for name, crit in CANONICAL.items():
        cand = crit.value(final_nll, k, n, k_eff) if (n and final_nll is not None) else None
        refv = _reference_score(ref, crit)
        criteria_scores[name] = {
            "candidate": cand,
            "reference": refv,
            "delta": (cand - refv) if (cand is not None and refv is not None) else None,
        }

    # The GATE runs under the criterion the server actually ranked models by, learned from the
    # `criterion` the scored fit reports (docs/RATIONALE.md). An unreadable or absent label falls
    # back to canonical BIC, which is also what the default reports.
    reported_criterion = final.get("criterion") if isinstance(final, dict) else None
    criterion = Criterion.from_name(reported_criterion)
    score_ref = _reference_score(ref, criterion) if criterion is not None else bic_ref
    criterion_note = None
    if criterion is not None and score_ref is None and bic_ref is not None:
        # The criterion has no value for the TRUE model — aicc is undefined at n <= k+1, tic needs
        # a k_eff the reference was written without — so it cannot grade against it either: fall
        # back to canonical BIC for BOTH sides and say so (docs/RATIONALE.md).
        criterion_note = (
            f"{criterion.name} has no value for the reference "
            f"(k={ref.get('k')}, n={ref.get('n')}, k_eff={ref.get('k_eff')}) — "
            "graded by canonical bic"
        )
        criterion, score_ref = None, bic_ref
    score_cand = (
        criterion.value(final_nll, k, n, k_eff)
        if (criterion is not None and n and final_nll is not None)
        else bic_cand
    )
    delta_score = (
        (score_cand - score_ref) if (score_cand is not None and score_ref is not None) else None
    )
    # The reference is scored on the problem's measurement table, one point per row, and so is a
    # candidate on the same table. A different n means the candidate was scored on other data, and
    # no Δ between the two means anything: withheld, as for the held-out set's own n_mismatch, so
    # the run fails. A null delivery could list extra measurements its model reproduces (#1).
    ref_n = ref.get("n") if isinstance(ref, dict) else None
    n_mismatch = bool(n) and isinstance(ref_n, (int, float)) and n != ref_n and not reference_free
    if n_mismatch:
        delta_bic = delta_score = None
        for row in criteria_scores.values():
            row["delta"] = None
    # One threshold for both: every criterion here is on the same deviance scale (2·Δnll), which
    # is what lets the benchmark's evidence bar carry over unchanged when the penalty is swapped.
    score_pass = delta_score is not None and delta_score <= scenario.bic_threshold
    success = score_pass and not sigma_implausible
    # What the gate COMPUTED under, for the verdict text — "bic" whenever no other criterion was
    # readable or usable. Deliberately NOT what the run RECORDS as its arm (`reported_criterion`
    # below): a run that died before any fit has no label to report, and calling that "bic" would
    # file an aborted AICc run under the BIC arm.
    criterion_label = criterion.name if criterion is not None else "bic"
    # χ² goodness-of-fit vs the known noise (reported, not gating).
    dof = max(0, n - k)
    p_gof = chi2_gof_pvalue(chi2, dof)
    # REFERENCE-FREE: no true model, so every Δ above is None and `score_pass` can never hold. The
    # bar is the absolute adequacy test the scenario declares instead; an unassessable verdict is NOT
    # a pass, the same conservatism as a missing reference.json on a reference-based example.
    adequacy = None
    if reference_free:
        adequacy = _reference_free_adequacy(
            scenario.adequacy, noise_sigma_report, p_gof, residual_structure
        )
        success = final_nll is not None and not sigma_implausible and adequacy["passed"] is True

    ctx = JudgeContext(
        call=call,
        baseline=baseline,
        final_nll=final_nll,
        structure=structure,
        new_species=new_species,
        delta=delta,
        n=n,
        k=k,
        chi2=chi2,
    )
    diag_lines: list[str] = []
    diag_fields: dict = {}
    if scenario.diagnostics is not None:
        try:
            diag_lines, diag_fields = await scenario.diagnostics(ctx)
        except Exception as exc:  # diagnostics are report-only; never fail the run on them
            diag_lines = [f"(diagnostics error: {type(exc).__name__}: {exc})"]

    print("\n" + "=" * 70)
    print(f"RESULT — {scenario.name}")
    print("=" * 70)
    print(f"  reactions now:        {[r.get('id') for r in reactions]}")
    for line in diag_lines:
        print(f"  {line}")
    print(f"  new species:          {new_species or 'none'}")
    _bl = f"{baseline.nll:.6g}" if baseline.nll is not None else "n/a (zero-shot)"
    _dl = f"{delta:.6g}" if delta is not None else "n/a"
    _nll = f"{final_nll:.6g}" if final_nll is not None else "n/a (no valid fit)"
    _keff = f"  k_eff={k_eff:.3g}" if isinstance(k_eff, (int, float)) else ""
    print(
        f"  fit: nll={_nll}  chi2={chi2:.6g}  k={k}{_keff}  n={n}  (baseline nll {_bl}, Δnll {_dl})"
        if chi2 is not None
        else f"  fit: nll={_nll}  k={k}{_keff}  n={n}  (baseline nll {_bl}, Δnll {_dl})"
    )
    # `ljust(5) + " "` keeps the historic "BIC:  candidate" column (two spaces) byte-identical
    # while a longer label just takes the one space it needs.
    _label = criterion_label.upper()
    _col = f"{_label}:".ljust(5) + " "
    if criterion_note:
        print(f"  note: {criterion_note}")
    if reference_free:
        # No bar to set the score against: the candidate's own numbers are the headline, comparable
        # only across runs on this same problem (lower is better).
        if score_cand is not None:
            print(f"  {_col}candidate {score_cand:.6g}  (reference-free: no true model, no bar)")
            if criterion_label != "bic" and bic_cand is not None:
                print(f"  BIC:  candidate {bic_cand:.6g}  (canonical, recorded)")
        else:
            print(f"  {_col}unavailable (no valid fit, n={n})")
        _adequate = {True: "adequate", False: "INADEQUATE", None: "not assessable"}
        print(
            f"  {'adequacy (' + str(adequacy['test']) + '):':<21} "
            f"{_adequate[adequacy['passed']]} — {adequacy['reason']}"
        )
    elif delta_score is not None:
        print(
            f"  {_col}candidate {score_cand:.6g} | true-model {score_ref:.6g} | "
            f"Δ{_label} {delta_score:+.6g}  (accept if <= {scenario.bic_threshold:g})"
        )
    else:
        # A reference that parses but lacks nll/k/n makes the reference score None while `ref` is a
        # truthy dict — report that as "malformed", not "ok", so the real cause of the missing gate
        # shows.
        ref_state = "missing" if not ref else ("ok" if score_ref is not None else "malformed")
        if n_mismatch:
            ref_state = f"scored on n={ref_n}"
        print(f"  {_col}unavailable (reference {ref_state}, n={n})")
    # Canonical BIC alongside whenever it is NOT the gate — recorded for every run
    # (docs/RATIONALE.md), and worth seeing next to the score that actually decided.
    if criterion_label != "bic" and delta_bic is not None:
        print(
            f"  BIC:  candidate {bic_cand:.6g} | true-model {bic_ref:.6g} | "
            f"ΔBIC {delta_bic:+.6g}  (recorded, not the gate)"
        )
    if p_gof is not None:
        print(
            f"  chi2 GoF:             p={p_gof:.3g} (dof={dof}) "
            f"— {'consistent with noise' if p_gof >= 0.05 else 'worse than noise'}"
        )
    if validation:
        # Printed next to ΔBIC because it is the same shape of comparison against the same true
        # model — on data neither side was fitted to.
        # Report-only; it is not part of the verdict.
        if reference_free and isinstance(validation.get("nll"), (int, float)):
            # No reference to subtract, so the ABSOLUTE held-out numbers. Still report-only.
            _vchi2 = validation.get("chi2")
            print(
                f"  held-out (validation): nll {validation['nll']:.6g}"
                + (f" | chi2 {_vchi2:.4g}" if isinstance(_vchi2, (int, float)) else "")
                + f"  over {validation.get('n')} point(s) in "
                f"{validation.get('n_experiments')} unseen experiment(s)  [report-only]"
            )
        elif validation.get("delta_nll") is not None:
            print(
                f"  held-out (validation): nll {validation['nll']:.6g} | true-model "
                f"{validation['reference_nll']:.6g} | Δnll "
                f"{validation['delta_nll']:+.6g}  over {validation['n']} point(s) in "
                f"{validation.get('n_experiments')} unseen experiment(s)  [report-only]"
            )
            # Explicitly non-None AND non-zero on the reference side: the ratio below divides by
            # it, so zero has to be excluded on purpose rather than by the accident of a
            # truthiness test that reads like a None check.
            _vchi2, _rchi2 = validation.get("chi2"), validation.get("reference_chi2")
            if _vchi2 is not None and _rchi2 is not None and _rchi2 != 0:
                # Ratio, not a p-value: the truth's own held-out reduced chi2 on fujita is 36.5, so
                # an absolute goodness-of-fit test rejects both sides (see scoring._validation_fields).
                print(
                    f"  held-out chi2:        {_vchi2:.4g} vs true-model {_rchi2:.4g} "
                    f"(x{_vchi2 / _rchi2:.2f})"
                )
        else:
            print(
                f"  held-out (validation): {validation['status']}"
                + (f" — {validation['reason']}" if validation.get("reason") else "")
                + (
                    "  (held-out n differs from the reference's — the difference is withheld)"
                    if validation.get("n_mismatch")
                    else ""
                )
                + (
                    "  (the reference block and the score are of different kinds — the "
                    "difference is withheld)"
                    if validation.get("mode_mismatch")
                    else ""
                )
            )
    # Server-side breakdown of the heavy compute: how much of the tool time was the one-time
    # AMICI compile vs the recurring pyPESTO fit (best-effort — a server without the tool, or
    # a failed call, just leaves these None). Fetched BEFORE the clock snapshot below so this
    # probe is counted consistently in BOTH compute_seconds and tool_calls (not one but not
    # the other), keeping the two metrics comparable.
    compile_s = fit_s = n_compiles = n_fits = None
    try:
        timing = await call("get_run_timing")
        if isinstance(timing, dict):
            compile_s = timing.get("compile_seconds")
            fit_s = timing.get("fit_seconds")
            n_compiles = timing.get("n_compiles")
            n_fits = timing.get("n_fits")
    except Exception:
        pass
    # Wall-clock split: time inside MCP tool calls (dominated by the AMICI compile +
    # pyPESTO fits + SQL) vs LLM+overhead (the remainder — inference + orchestration).
    wall = time.monotonic() - t0
    compute_s = clock.seconds
    llm_s = max(0.0, wall - compute_s)
    if clock.calls:
        breakdown = ""
        if compile_s is not None and fit_s is not None:
            breakdown = (
                f" [compile {compile_s:.0f}s ×{n_compiles} + fit {fit_s:.0f}s ×{n_fits}"
                f" + other {max(0.0, compute_s - compile_s - fit_s):.0f}s]"
            )
        print(
            f"  time:                 total {wall:.0f}s | compute {compute_s:.0f}s "
            f"({clock.calls} tool calls){breakdown} | LLM+overhead {llm_s:.0f}s"
        )
    if errors:
        print(f"  rejected tool calls:  {len(errors)} (see get_errors)")
    ant_diff = diff.get("model_diff") if isinstance(diff, dict) else None
    if ant_diff:
        print("  model diff (v0 -> HEAD):")
        for line in str(ant_diff).splitlines():
            print("    " + line)
    if reference_free:
        if final_nll is None:
            verdict = "no valid fit (the model was never built or produced a non-finite objective)."
        elif adequacy["passed"] is False:
            verdict = f"inadequate by its declared {adequacy['test']} test: {adequacy['reason']}."
        elif sigma_implausible:
            verdict = (
                f"the estimated noise sigma is {_implausible_sigma_text(noise_sigma_flag)} — an "
                "implausible noise model, so the fit does not count."
            )
        elif adequacy["passed"] is None:
            verdict = (
                f"adequacy not assessable by its declared {adequacy['test']} test "
                f"({adequacy['reason']}); an unassessed fit is not a pass."
            )
        else:
            verdict = (
                f"adequate by its declared {adequacy['test']} test ({adequacy['reason']}) — an "
                "absolute bar, with no true model to compare against."
            )
    elif success:
        verdict = (
            f"not rejectable vs the true model (Δ{criterion_label} {delta_score:+.3g} <= "
            f"{scenario.bic_threshold:g}); the repair explains the data as well as the truth."
        )
    elif final_nll is None:
        verdict = "no valid fit (the model was never built or produced a non-finite objective)."
    elif n_mismatch:
        verdict = (
            f"scored on n={n} data points, the true model on n={ref_n}: the two scores are not "
            f"comparable, so there is no Δ{criterion_label}."
        )
    elif delta_score is None:
        verdict = (
            f"could not evaluate {criterion_label} (missing reference.json or no data points)."
        )
    elif score_pass and sigma_implausible:
        verdict = (
            f"{criterion_label} passes (Δ{criterion_label} {delta_score:+.3g}) but the "
            f"estimated noise sigma is {_implausible_sigma_text(noise_sigma_flag)} — the good "
            "score rests on an implausible noise model, so it does not count."
        )
    else:
        verdict = (
            f"rejectable: Δ{criterion_label} {delta_score:+.3g} > "
            f"{scenario.bic_threshold:g} — the fit is decisively worse than the true "
            "model."
        )
    print(f"\n  {'SUCCESS' if success else 'NOT FIXED'}: {verdict}")
    print("=" * 70)

    result = {
        **meta,
        "scenario": scenario.name,
        "baseline_nll": baseline.nll,
        "final_nll": final_nll,
        "delta": delta,
        "chi2": chi2,
        "k": k,
        # k's two addends, so a k above the estimated-row count is auditable after the fact. On a
        # null delivery the first is the reconcile's count; `delivery.uncharged` says how it differs.
        "k_estimated": k_estimated,
        "hardcoded_constants": hardcoded,
        # The scored model's parameter VECTOR, not just its count. Every criterion here prices k
        # parameters and none of them recorded what those parameters came out as, so reading a
        # finished run's fit back -- to compare two runs' constants, to seed another fit, to say
        # what a distilled rate law actually claims -- meant re-fitting it.
        "fitted_parameters": _fitted_parameters(params),
        "n": n,
        # n differs from the reference's: every Δ below is withheld and the run fails.
        "n_mismatch": n_mismatch,
        "k_eff": k_eff,
        "bic": bic_cand,
        "bic_reference": bic_ref,
        "delta_bic": delta_bic,
        # None when reference-free: there is no Δ for it to bound.
        "bic_threshold": None if reference_free else scenario.bic_threshold,
        # Candidate, reference and delta under EVERY named criterion — the flat `bic*` triple and
        # the `criterion*` quadruple above are the two rows of this table that the CSV/W&B/span
        # allowlists flatten; the rest is here for reading a run under a criterion it was not run
        # or graded with.
        "criteria": criteria_scores,
        # The GATE's own numbers, under the criterion the server ranked by and read off the
        # scored fit rather than re-derived here (docs/RATIONALE.md). The `bic*` fields above stay
        # CANONICAL BIC for every run; under the default criterion the two sets are equal.
        # `criterion` is what the SERVER reported — `None` when the run never produced a fit to
        # report one, so an aborted run is not filed under the wrong arm — and `criterion_note`
        # says when the gate had to grade under something else.
        "criterion": reported_criterion,
        "criterion_note": criterion_note,
        "criterion_value": score_cand,
        "criterion_reference": score_ref,
        "delta_criterion": delta_score,
        "chi2_dof": dof,
        "chi2_gof_pvalue": p_gof,
        # Which bar decided `success`: "reference" (Δcriterion against the true model, above) or
        # "reference_free" (the declared absolute adequacy test below). The two passes are
        # different claims, so a sweep must group by this before it pools a pass rate.
        "judge_mode": judge_mode,
        # The reference-free bar: the declared test, its verdict and why. None on a reference run.
        "adequacy": adequacy,
        "residual_structure": residual_structure,
        # HELD-OUT prediction: this model's likelihood on measurements it was never fitted to,
        # against the true model's. REPORT-ONLY — deliberately not in the `success` gate above.
        # `None` on an example that ships no held-out data.
        "validation": validation,
        # The causal_constraints arm's refusals, first satisfying version and behaviour at the fit;
        # None off the arm.
        "causal_constraints": causal_constraints,
        # Noise-sigma plausibility of the scored model: GATES success (an inflated/collapsed sigma
        # disqualifies an otherwise-passing score; see the gate above) but never shown to the agent.
        "noise_sigma_report": noise_sigma_report,
        "noise_sigma_flag": noise_sigma_flag,
        # which comparison produced the flag: "reference" (vs the true model's own
        # fitted sigma, preferred) or "data_std" (the legacy marginal-std band).
        "noise_sigma_basis": sigma_basis,
        "reference": ref,
        "reactions": [r.get("id") for r in reactions],
        "species": sorted(current_species),
        "new_species": new_species,
        "success": success,
        "head_version": head,
        "n_errors": len(errors) if isinstance(errors, list) else 0,
        "wall_seconds": round(wall, 1),
        "compute_seconds": round(compute_s, 1),
        "llm_seconds": round(llm_s, 1),
        # Server-side split of the heavy compute (None if the server didn't report it).
        "compile_seconds": compile_s,
        "fit_seconds": fit_s,
        "n_compiles": n_compiles,
        "n_fits": n_fits,
        "tool_calls": clock.calls,
        # Which tools, not just how many. Answers "did the agent ever call X?" from result.json
        # alone -- no Phoenix collector, no `.slurm_scratch` (which is deleted on completion).
        "tool_call_counts": dict(sorted(clock.by_tool.items(), key=lambda kv: -kv[1])),
        # Rejected edits are SUCCESSFUL tool calls returning applied=false, so
        # n_tool_errors never sees them (see _ToolClock.note_result).
        "edit_rejects": clock.reject_summary(),
        "run_ended_at": _now_iso(),
        **diag_fields,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"  (machine-readable result -> {(work / 'result.json').relative_to(scenario.example_dir.parent)})"
    )
    return (0 if success else 1), result
