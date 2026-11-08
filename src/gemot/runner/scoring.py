"""Projections of the server's fit reports into the judge's verdict fields.

Three families, all pure functions over a report payload:

* the compact residual-structure block ``result.json`` carries,
* the noise-sigma plausibility verdicts — is the estimated sigma inflated to absorb misfit, or
  collapsed to overfit the noise? The gate is the server's own data-std band; the comparison
  against the reference model's fitted sigmas is recorded beside it and decides nothing — and
* the absolute adequacy verdict that is a reference-free example's bar, where there is no true
  model to compare against at all.

Imports ``petab_mcp.constants`` so the session's adequacy gate and this judge gate share ONE
threshold — but nothing from ``mcp``, so a test of these projections needs no MCP client.
"""

from __future__ import annotations


# The collapse threshold lives in petab_mcp.constants so the session's adequacy gate and this
# judge gate share ONE value (re-exported under the old name for callers/tests).
from petab_mcp.constants import NOISE_SIGMA_COLLAPSE_FRACTION as _NOISE_SIGMA_COLLAPSE_FRACTION
from petab_mcp.constants import (
    NOISE_SIGMA_BAND_ALPHA as _NOISE_SIGMA_BAND_ALPHA,
    NOISE_SIGMA_BAND_MIN_DF as _NOISE_SIGMA_BAND_MIN_DF,
    NOISE_SIGMA_REFERENCE_FACTOR as _NOISE_SIGMA_REFERENCE_FACTOR,
)

from ..scenario import (
    ADEQUACY_CHI2,
    ADEQUACY_REPLICATE_LOF,
    ADEQUACY_RESIDUAL_STRUCTURE,
    REFERENCE_FREE_GOF_ALPHA,
)

__all__ = [
    "_NOISE_SIGMA_COLLAPSE_FRACTION",
    "_NOISE_SIGMA_BAND_ALPHA",
    "_NOISE_SIGMA_BAND_MIN_DF",
    "_NOISE_SIGMA_REFERENCE_FACTOR",
    "REFERENCE_FREE_GOF_ALPHA",
    "_noise_sigma_flag",
    "_reference_free_adequacy",
    "_reference_sigma_flag",
    "_residual_structure_fields",
    "_scale_coupled_sigmas",
    "_validation_fields",
]


def _validation_fields(
    payload, reference, *, asked: bool = False, waited_s: float | None = None
) -> dict | None:
    """The HELD-OUT (validation) block for result.json, or ``None`` when this run has no such check.

    ``payload`` is what ``evaluate_validation_data`` returned (:mod:`petab_mcp.validation_data`);
    ``reference`` is ``reference.json``, whose ``validation`` block holds the same number for the
    true model. The figure worth reading is ``delta_nll`` = candidate − reference: POSITIVE means
    the candidate predicts the held-out conditions worse than the true model does, with no
    complexity term, because neither side spent a parameter on this data.

    REPORT-ONLY, deliberately rather than provisionally: a threshold would need a distribution of
    this number across runs and examples, and recording it is how that gets collected.

    Every failure keeps its own ``status`` rather than collapsing to a null ``nll``.
    ``unavailable`` means the tool was absent or unreadable. ``timed_out`` means the call ran out of
    time after ``waited_s`` seconds, which a sweep must not read as a missing tool: a transfer set
    refits the model, and all six lucarelli runs of one sweep recorded ``unavailable`` for a call
    cut at 900 s (gemot#57). Neither is a statement about the model. ``asked`` says the judge DID
    attempt the call, which separates a run with no held-out set (``None``) from one whose check was
    attempted and failed while the reference had no bar yet — a silent ``null`` there is
    indistinguishable from a clean skip. The rest come from the scorer and are statements about the
    run. ``n_mismatch`` and ``mode_mismatch`` guard the subtraction itself —
    the two sides are computed at different times from different files, and a score of one KIND
    must not be subtracted from a score of another — and withhold ``delta_nll`` when they fire.

    NO goodness-of-fit p-value, even though the held-out noise is known: it would be a constant.
    The true model's own held-out reduced chi2 on fujita is 36.5, so the test rejects the truth as
    decisively as anything and every run would record ``p = 0`` — a property of the DATASET, which
    belongs in BENCHMARK_NOTES. ``reference_chi2`` is carried instead.
    """
    ref_block = (reference or {}).get("validation") if isinstance(reference, dict) else None
    ref_block = ref_block if isinstance(ref_block, dict) else {}
    # The REFERENCE side's own status, carried through so a refit whose held-out scoring FAILED
    # (`status: error`, written by `petab_agent_bench.reference.write_reference_from_problem`) is
    # distinguishable from a reference that simply predates the feature. Both leave
    # `reference_nll` blank, and without this the two are byte-identical in the history CSV —
    # which would make the guarded refit's whole point ("record the failure rather than drop it")
    # invisible per-run.
    # No default: a block that does not SAY it succeeded is reported as `None`, not as "ok".
    # Folding "the block is silent" into the affirmative case would defeat the point of carrying
    # the status at all, which is to separate "the refit's held-out scoring failed" from "no bar
    # exists". Every block `validation_data.score_problem` writes carries one.
    ref_status = ref_block.get("status") if ref_block else None
    # EVERY value is gated on the status, `n` included. Reading `n` from a block whose likelihood
    # we just refused to read would let `n_mismatch` assert "the two sides counted different
    # numbers of held-out points" on the authority of a reference declared unusable one line
    # above — a positive claim derived from data we do not trust.
    trusted = ref_status == "ok"
    ref_nll = ref_block.get("nll") if trusted else None
    ref_n = ref_block.get("n") if trusted else None
    ref_chi2 = ref_block.get("chi2") if trusted else None
    # A block written before modes existed is a fixed-θ block; the payload always says its own.
    ref_mode = (ref_block.get("mode") or "fixed") if ref_block else None

    if not isinstance(payload, dict) or not payload.get("status"):
        # No held-out check ran. Distinguished from a check that ran and failed: an example with no
        # validation set must not look like one whose model would not integrate.
        # `asked` covers the third case — the judge DID call and got nothing back, with the
        # reference not yet refit — which must not read as a clean skip.
        if not ref_block and not asked:
            return None
        # `reference_chi2` rides along with the other reference fields: the block's SHAPE must not
        # depend on whether the tool answered, or every consumer special-cases the failure.
        block = {
            "status": "unavailable",
            "reason": (
                "the server reported no held-out score (no evaluate_validation_data "
                "tool, or the call failed)"
            ),
            "mode": ref_mode,
            "reference_status": ref_status,
            "reference_nll": ref_nll,
            "reference_chi2": ref_chi2,
            "reference_n": ref_n,
        }
        if waited_s is not None:
            block["status"] = "timed_out"
            block["reason"] = f"the held-out score did not return within {waited_s:.0f} s"
            block["waited_s"] = round(float(waited_s), 1)
        return block

    mode = str(payload.get("mode") or "fixed")
    out = {
        "status": str(payload["status"]),
        "mode": mode,
        "nll": payload.get("nll"),
        "n": payload.get("n"),
        "chi2": payload.get("chi2"),
        # The server keys a held-out set by experiment since its move onto PEtab v2. No fallback
        # to the old `n_conditions`: the pinned server never sends it, and the reference blocks
        # that still carry it are never read for the count.
        "n_experiments": payload.get("n_experiments"),
        "reference_status": ref_status,
        "reference_nll": ref_nll,
        "reference_chi2": ref_chi2,
        "reference_n": ref_n,
    }
    if payload.get("reason"):
        out["reason"] = payload["reason"]

    nll, n = payload.get("nll"), payload.get("n")
    if ref_mode is not None and ref_mode != mode:
        out["mode_mismatch"] = True
    elif isinstance(ref_n, int) and isinstance(n, int) and ref_n != n:
        out["n_mismatch"] = True
    elif isinstance(nll, (int, float)) and isinstance(ref_nll, (int, float)):
        out["delta_nll"] = float(nll) - float(ref_nll)
    return out


def _residual_structure_fields(diagnosis, rollup=None) -> dict | None:
    """A compact, bounded residual-structure block for result.json, or ``None`` when neither half
    is available (no simulation / model-less run).

    TWO inputs because the two live in two places: ``rollup`` is the fit's own
    ``residual_structure`` verdict (read here off ``get_fit_summary``; see
    ``data_access.residual_structure_rollup`` for which surfaces carry it), and ``diagnosis`` is the
    ``get_fit_diagnosis`` payload the per-observable rows come from — its residual section, since
    nothing here reads its parameter half.

    Keeps the rolled-up verdict (which observables show significant sign-test bias or runs-test
    time trend, and whether the tests were underpowered) plus one row per observable with the
    two p-values and the bias magnitude — enough to analyse a sweep without the per-point data.

    ``rollup_status`` and ``rows_status`` say which of the two halves was actually READ, because they
    arrive from two different calls now and either can be missing. When the verdict is absent — a
    non-dict, an ``{ok: false}`` envelope, a payload without the key — ``flagged_observables`` and
    ``low_power`` are OMITTED rather than defaulted: defaulting them wrote "no observable flagged, and
    the tests were powered" into result.json for a run whose verdict was never read, and a sweep
    grouped on those columns cannot tell that from a genuine clean pass. Absent means UNKNOWN, and
    ``rows_status`` is the same distinction for the rows, whose ``[]`` is otherwise identical to a
    genuinely empty diagnosis.

    ``replicate_lack_of_fit`` is carried through verbatim: it is already the bounded projection the
    fit result rides on (a verdict plus at most REPLICATE_LOF_MAX_REPORTED_POINTS named points), and
    without it a sweep cannot tell whether the arm FLAGGED, PASSED (and on how many lack-of-fit
    degrees of freedom — ``weak``) or never ran. Without it a sweep cannot tell those apart
    from its result.json — establishing that the arm had fired at all meant going to the Phoenix
    trace, which does not scale to a series. ``metrics._flatten_replicate_lof`` takes it the last
    step, into the flat ``lof_*`` columns of the history CSV a sweep is actually grouped by.
    """
    per = diagnosis.get("per_observable") if isinstance(diagnosis, dict) else None
    if not isinstance(rollup, dict) and not isinstance(per, list):
        return None
    # A truthy NON-dict `residual_structure` (a status string from a degraded payload) satisfies
    # nothing here: the guard above is satisfied by `per` alone, and `.get` on a string raised
    # AttributeError — which judge.py catches by setting the whole block to None, losing the
    # per-observable rows too. This is the projection that makes a run readable from result.json; it
    # must degrade, not vanish.
    # A verdict is a dict that CARRIES one: every rollup does, degraded ones included (see
    # data_access._unavailable_structure), so an `{ok: false}` envelope or any other dict-shaped
    # non-verdict reads as unavailable rather than as an empty flagged list.
    read = isinstance(rollup, dict) and "flagged_observables" in rollup
    per_obs = []
    for o in per or []:
        if not isinstance(o, dict):
            continue
        st = o.get("sign_test") or {}
        rt = o.get("runs_test") or {}
        per_obs.append(
            {
                "observable_id": o.get("observable_id"),
                "sign_test_p": st.get("p"),
                "runs_test_p": rt.get("p"),
                "mean_weighted_residual": o.get("mean_weighted_residual"),
                "low_power": bool(o.get("structure_low_power")),
            }
        )
    out = {
        "rollup_status": "read" if read else "unavailable",
        # "truncated" is its own status: since 2026-08-16 fit_diagnosis caps per_observable on
        # large problems, and a sweep aggregate over a worst-chi2-biased subsample that calls
        # itself "read" is the silent-truncation failure this field exists to prevent.
        "rows_status": (
            "truncated"
            if isinstance(per, list) and (diagnosis or {}).get("omitted")
            else "read"
            if isinstance(per, list)
            else "unavailable"
        ),
        **(
            {"rows_omitted": (diagnosis or {}).get("omitted")}
            if isinstance(per, list) and (diagnosis or {}).get("omitted")
            else {}
        ),
        "per_observable": per_obs,
    }
    if read:
        out["flagged_observables"] = rollup.get("flagged_observables", [])
        out["low_power"] = bool(rollup.get("low_power"))
        lof = rollup.get("replicate_lack_of_fit")
        if isinstance(lof, dict):
            out["replicate_lack_of_fit"] = lof
    return out


def _scale_coupled_sigmas(report) -> list[str]:
    """The sigmas the server marks ``scale_coupled`` (``reconcile.sigma_scale_coupled``), sorted:
    those whose raw value is not comparable across models, so :func:`_reference_sigma_flag`
    declines its basis on any of them."""
    if not isinstance(report, dict):
        return []
    return sorted(
        str(pid) for pid, e in report.items() if isinstance(e, dict) and e.get("scale_coupled")
    )


def _reference_sigma_flag(report, reference, k=None, n=None) -> dict | None:
    """Plausibility of the candidate's ESTIMATED sigmas against the TRUE model's OWN fitted sigmas
    (``reference.json``'s ``noise_sigma``), or ``None`` when the reference doesn't record them.

    RECORDED in ``result.json`` (``noise_sigma_reference``), never a gate. It was the judge's
    preferred gate from #118 (the marginal data std conflates variance the model EXPLAINS with
    variance it cannot, so legitimate sigmas span ~0.0004-0.59 of it), until gemot#135: the
    reference is the published model fitted to real data, so its sigma is that model's misfit, not
    the noise; reference data errors failed honest runs (fiedler, isensee gemot#104, schwen and
    lucarelli PEtabAgentBench#53); and a rule the agent could not see made the finish gate promise
    a version the judge then passed over. With the noise model read-only and sigma fitted by ML, no
    sigma beats the ML one, so BIC already charges what this ratio measures.

    How far from 1 is too far depends on HOW MUCH DATA the sigma was estimated from, which the fixed
    tolerance factor this used to apply alone could not know. A fitted sigma is an ML estimate: its
    squared value is a mean of squared residuals, so ``sigma_hat^2 / sigma^2 ~ chi2_d / d`` and the
    two sides' ratio is F-distributed — ``(fitted/reference)^2 ~ F(d_candidate, d_reference)``. A
    sigma is flagged only outside the WIDER of that distribution's central
    ``1 - NOISE_SIGMA_BAND_ALPHA`` interval (Bonferroni-split across the sigmas tested, since
    flagging on ANY of several is a multiple comparison) and the fixed factor — see
    :func:`_sigma_ratio_band` for why the union rather than the F band alone.

    The fixed factor of 3 was, in effect, the 99% interval for a sigma with about 7 residual degrees
    of freedom, applied to every sigma regardless of its own. That is far too tight below 7:
    brannmark's ``sigmaYDosR`` is pooled over SEVEN dose-response points, one per observable, so at
    its real df a ratio of 0.25 is a ~1-2% event, and the fixed band called it certain gaming and
    discarded the run (9285a352, ΔBIC −21.65). Widening there is what this fixes; the factor stays as
    the floor on tolerance, so nothing that passed before fails now.

    One assumption is worth naming: ``F`` treats the two sigmas as independent, and they are not —
    both are fitted to the SAME measurements, so their errors are correlated and the true null is
    tighter than the F. The band is therefore CONSERVATIVE: the realised false-flag rate is below
    the nominal ``alpha`` recorded in ``result.json``, not above it. For a gate whose job is to
    catch gaming without discarding honest runs, erring that way is the right direction, but the
    recorded alpha should be read as a bound rather than a rate.

    ``d`` is the group's share of the model's residual degrees of freedom — its point count scaled by
    ``1 - k/n``, the same crude attribution the replicate lack-of-fit arm makes when it charges each
    design point a share of the structural parameters. Where the count is unknown, or the share
    leaves less than ``NOISE_SIGMA_BAND_MIN_DF``, the sampling band means nothing and the FIXED
    factor is applied on its own — the gate stays on, exactly as strict as it was before this
    existed. Every sigma gets a verdict either way; ``per_sigma`` records which band produced it
    (``df`` is ``None`` where the count was unknown), so a pass on a tiny group is auditable.

    Returns ``{inflated, collapsed, max_ratio, min_ratio, alpha, per_sigma}`` where the ratios are
    ``fitted / reference`` per sigma and ``per_sigma`` carries each one's ratio, df, band and
    whether it compared effective or raw sigmas.
    """
    if not isinstance(report, dict) or not report:
        return None
    # A sigma that shares its noiseFormula with an ESTIMATED gain is declined here. To be precise
    # about WHY, because it is not that the quantity is meaningless:
    #
    #   * The RAW sigma is model-specific. Only the product `gain * sigma` is identified, so a model
    #     at different internal units reaches the SAME noise with a proportionally different sigma.
    #     fiedler: reference sigma_pMek = 5.8e-4 against gel gains of 77.7 .. 999.99.
    #   * The PRODUCT is perfectly comparable -- it is the noise sd in measurement units, and both
    #     models fit the same measurements, so two good fits should agree on it. 404.2 * 5.8e-4 =
    #     0.235, against a pMek data std of 1.35: plausible, and nothing like the 179x "inflation"
    #     the raw ratio reports.
    #
    # So this is a DATA-AVAILABILITY limit, not an impossibility. Neither side currently carries the
    # product: the candidate entry's `effective_sigma` has the amplitude divided out by
    # `sigma_effective_value` (deliberately -- the gain cancels from the data-driven sd/spread ratio,
    # which is why that basis is unaffected), and reference.json records only the raw sigma, with
    # `noise_sigma_effective` equal to it. Comparing effective-vs-effective therefore reduces to
    # raw-vs-raw and changes nothing.
    #
    # The durable fix is to record the gain-inclusive sd in reference.json at refit time. It cannot
    # be reconstructed generally from the shipped nominals: they ARE the reference fit for fiedler
    # (verified -- both sigmas match reference.json exactly) but differ on most examples (alkan,
    # isensee, oliveira, schwen, blasi...), so reading them would silently compare against the wrong
    # model. Until then, decline the whole comparison as the unpairable-sigma case below does. The
    # data-driven band, the judge's gate, was right all along here: it never
    # flagged any of the 116 runs scored on it, while this basis silently failed 17 fiedler runs that
    # had BEATEN the reference bar. See reconcile.sigma_scale_coupled, which since PEtab-MCP#243 also
    # counts a co-parameter written into the noiseFormula itself: isensee's scales are, and while the
    # flag read only `noiseParameters` this basis refused isensee r2's v13-v16 (dBIC -12 to -21)
    # for a raw rho the split between scale and sigma had moved (gemot#104).
    if _scale_coupled_sigmas(report):
        return None
    # Prefer the EFFECTIVE sd on BOTH sides: for a composite/proportional noiseFormula the raw
    # parameter is not comparable across models (fiedler's sd is per-gel scale * sigma, so a
    # candidate at a different amplitude trades scale against sigma and reaches an IDENTICAL sd
    # with a raw sigma several-fold off — judged raw, it fails for matching the truth).
    #
    # The side is chosen PER SIGMA, raw-vs-raw wherever either side has no effective value. Not
    # once for the whole reference: one sigma without an effective value then left its pair
    # mixed or missing, and the unpairable-sigma guard below switched the WHOLE basis off. That
    # was bachmann (#3): `sd_STAT5_au` is shared by observables whose noiseFormulas evaluate to
    # different sds, so `sigma_effective_value` is None for it on both sides, and every bachmann
    # run was silently judged on the data-std band. A reference written before
    # noise_sigma_effective existed lands on raw-vs-raw throughout, as it always did.
    reference = reference if isinstance(reference, dict) else {}
    raw_ref = reference.get("noise_sigma")
    raw_ref = raw_ref if isinstance(raw_ref, dict) else {}
    effective_ref = reference.get("noise_sigma_effective")
    effective_ref = effective_ref if isinstance(effective_ref, dict) else {}
    ref_ids = set(raw_ref) | set(effective_ref)
    if not ref_ids:
        return None
    # The reference is fit on the PRISTINE collection problem, whose parameter ids can differ from
    # the materialized ones an agent sees (perelson ships `sd_viral_load` but the collection calls
    # it `sd_task0_model0_perelson1_V`). When BOTH sides have exactly one sigma there is no
    # ambiguity about the pairing, so match them positionally rather than losing the comparison —
    # anything more than one-to-one stays id-matched, since a wrong pairing is worse than none.
    solo_pairing = len(report) == 1 and len(ref_ids) == 1
    solo_ref_id = next(iter(ref_ids)) if solo_pairing else None
    # Bonferroni: the verdict fires if ANY sigma is outside its band, so testing g of them at alpha
    # each would fail an honest run g times as often as the alpha claims. Counts the entries the
    # loop below actually TESTS — a non-dict entry it skips is not a comparison, and counting it
    # would split alpha further and widen every real sigma's band.
    alpha = _NOISE_SIGMA_BAND_ALPHA / max(1, sum(1 for e in report.values() if isinstance(e, dict)))
    hi = lo = None
    inflated = collapsed = False
    per_sigma: dict[str, dict] = {}
    for pid, entry in report.items():
        if not isinstance(entry, dict):
            continue
        ref_pid = solo_ref_id if solo_pairing else pid
        # Effective vs effective where BOTH sides carry one, else raw vs raw — never one of each.
        fitted, ref_value = entry.get("effective_sigma"), effective_ref.get(ref_pid)
        compared_on = "effective"
        if fitted is None or ref_value is None:
            fitted, ref_value = entry.get("fitted"), raw_ref.get(ref_pid)
            compared_on = "raw"
        try:
            fitted, ref_value = float(fitted), float(ref_value)
        except (TypeError, ValueError):
            # EVERY reported sigma must be pairable and usable. Comparing the rest would record a
            # max_ratio ~ 1 for a model whose other sigmas are far off. No record instead of a
            # partial one.
            return None
        if not (fitted > 0 and ref_value > 0):
            return None
        ratio = fitted / ref_value
        hi = ratio if hi is None else max(hi, ratio)
        lo = ratio if lo is None else min(lo, ratio)
        floor, ceiling, df_cand, df_ref = _sigma_ratio_band(
            entry.get("n_points"), k, n, reference, alpha
        )
        detail = {
            "ratio": ratio,
            # Which pair the ratio is of, so a raw-vs-raw fallback is visible in result.json.
            "compared_on": compared_on,
            "n_points": entry.get("n_points"),
            "band_low": floor,
            "band_high": ceiling,
            "df": df_cand,
            "df_reference": df_ref,
        }
        if ratio > ceiling:
            inflated = True
            detail["verdict"] = "inflated"
        elif ratio < floor:
            collapsed = True
            detail["verdict"] = "collapsed"
        else:
            detail["verdict"] = "ok"
        per_sigma[pid] = detail
    if hi is None:
        return None
    return {
        "inflated": inflated,
        "collapsed": collapsed,
        "max_ratio": hi,
        "min_ratio": lo,
        # The per-sigma detail behind the verdict: each ratio, the df it was judged at, and the
        # band that df implies. Written to result.json so a flag (or the absence of one on a
        # low-power sigma) can be audited without re-deriving any of it.
        "alpha": alpha,
        "per_sigma": per_sigma,
        # Same key names the data-std fallback emits, so a consumer of result.json's
        # noise_sigma_flag sees ONE shape regardless of which basis produced it.
        "max_fraction_of_data_std": _max_fraction_of_data_std(report),
        "min_fraction_of_data_std": _min_fraction_of_data_std(report),
    }


def _sigma_ratio_band(n_points, k, n, reference, alpha):
    """``(low, high, df_candidate, df_reference)`` — how far ``fitted/reference`` may stray before
    this sigma is called implausible.

    The WIDER of two bands, never the narrower:

    * the sampling band — the central ``1 - alpha`` interval of ``F(d_candidate, d_reference)``,
      where ``d = n_points * (1 - k/n)`` is the group's points less its share of the parameters the
      model spends (each side gets its own: a 22-parameter truth and a 12-parameter candidate do not
      leave the same freedom in the same points);
    * the fixed ``[1/FACTOR, FACTOR]`` tolerance this used to apply alone.

    Taking the union is the whole point. The sampling band is what a ratio could reach BY CHANCE if
    both models were the truth, and on a small group that is much wider than the fixed factor —
    which is the miscalibration that discarded brannmark run 9285a352. But it also SHRINKS as the
    group grows, and at a few hundred points it is tighter than ±20%, which would start failing
    honest near-misses: a good model that is not literally the truth has somewhat larger residuals,
    and that difference is real rather than sampling noise, so a sharp test rejects it. The fixed
    factor was chosen for exactly that reason and stays as the floor on tolerance.

    So this only ever RELAXES the old gate — never tightens it — and the relaxation is confined to
    sigmas whose df cannot support the old one.

    Tolerance is MONOTONE in df: fewer points can only widen the band, never narrow it. That is
    why the floor clamps the df the band is COMPUTED at rather than switching to the fixed factor
    below ``NOISE_SIGMA_BAND_MIN_DF``: a regime switch inverts the property — at brannmark's shape
    a FIVE-point sigma would be judged in [0.333, 3.0] while a seven-point one gets [0.154, 8.65],
    the thinner evidence drawing the harsher test. Clamping makes tolerance stop widening below
    the floor instead of collapsing. A missing point count is the one case with no df at all, and
    falls back to the fixed factor.
    """
    from scipy.stats import f as _f

    fixed = (1.0 / _NOISE_SIGMA_REFERENCE_FACTOR, float(_NOISE_SIGMA_REFERENCE_FACTOR))
    try:
        points = int(n_points)
        k_cand, n_cand = int(k), int(n)
        k_ref, n_ref = int(reference["k"]), int(reference["n"])
    except (TypeError, ValueError, KeyError):
        return (*fixed, None, None)  # unknown df: the fixed tolerance, as before
    if points <= 0 or n_cand <= 0 or n_ref <= 0:
        return (*fixed, None, None)
    df_cand = points * max(0.0, 1.0 - k_cand / n_cand)
    df_ref = points * max(0.0, 1.0 - k_ref / n_ref)
    # Clamped, not switched: at one or two degrees of freedom the band is so wide it says nothing,
    # and letting it keep widening would make the gate vanish on a two-point group. Bounding how
    # permissive it can get is a choice about the limit; the REPORTED df stays the real one.
    lo_df = max(df_cand, _NOISE_SIGMA_BAND_MIN_DF)
    hi_df = max(df_ref, _NOISE_SIGMA_BAND_MIN_DF)
    # The band is on the RATIO, the F quantiles are on its square.
    return (
        min(fixed[0], float(_f.ppf(alpha / 2, lo_df, hi_df)) ** 0.5),
        max(fixed[1], float(_f.ppf(1 - alpha / 2, lo_df, hi_df)) ** 0.5),
        df_cand,
        df_ref,
    )


def _fractions_of_data_std(report) -> list[float]:
    """Every numeric ``fraction_of_data_std`` in a noise_sigma_report (may be empty)."""
    out = []
    for entry in (report or {}).values():
        if isinstance(entry, dict):
            frac = entry.get("fraction_of_data_std")
            if isinstance(frac, (int, float)):
                out.append(float(frac))
    return out


def _max_fraction_of_data_std(report) -> float | None:
    fractions = _fractions_of_data_std(report)
    return max(fractions) if fractions else None


def _min_fraction_of_data_std(report) -> float | None:
    fractions = _fractions_of_data_std(report)
    return min(fractions) if fractions else None


def _noise_sigma_flag(report) -> dict | None:
    """The judge's noise-sigma gate: is any ESTIMATED noise sigma implausible relative to the data
    — INFLATED past its data-driven cap (absorbing model misfit) or COLLAPSED far below the data
    std (overfitting the noise for a suspiciously good score)? Returns
    ``{inflated, collapsed, max_fraction_of_data_std, min_fraction_of_data_std}`` or ``None`` when
    the problem has no estimated sigma.

    An inflated/collapsed sigma DISQUALIFIES an otherwise-passing score (see _judge). The verdict is
    the server's own (its predicates, below), the one every fit result and the finish gate show the
    agent (gemot#135). The full block is also written to result.json so a gated (or borderline) run
    is auditable. The collapse threshold
    ``_NOISE_SIGMA_COLLAPSE_FRACTION`` is a heuristic — tune it if a genuinely better, more flexible
    model (a legitimately small residual sigma) is being false-failed.
    """
    if not isinstance(report, dict) or not report:
        return None
    # Single source of truth for both predicates (shared with the session adequacy gate).
    from petab_mcp.fitting import noise_sigma_collapsed, noise_sigma_inflated

    inflated = noise_sigma_inflated(report)
    collapsed = noise_sigma_collapsed(report)
    hi = lo = None
    for e in report.values():
        if not isinstance(e, dict):
            continue
        frac = e.get("fraction_of_data_std")
        if isinstance(frac, (int, float)):
            hi = frac if hi is None else max(hi, frac)
            lo = frac if lo is None else min(lo, frac)
    return {
        "inflated": inflated,
        "collapsed": collapsed,
        "max_fraction_of_data_std": hi,
        "min_fraction_of_data_std": lo,
    }


def _reference_free_adequacy(
    test,
    noise_sigma_report,
    chi2_gof_pvalue,
    residual_structure,
    alpha: float = REFERENCE_FREE_GOF_ALPHA,
) -> dict:
    """Is the candidate adequate ON ITS OWN? The success bar of a reference-free example, which has
    no true model to be compared against, graded by the absolute goodness-of-fit ``test`` the
    scenario declares (``Scenario.adequacy``):

    * ``chi2`` — the residuals against the problem's own noise model. Valid only when the fit
      estimated no noise sigma, i.e. ``noise_sigma_report`` is an EMPTY dict: an ML sigma pins chi2
      near ``n``, so the test passes any model. A non-empty report makes the declared test invalid
      for this fit, which is reported as exactly that — never answered by another test instead.
    * ``replicate_lof`` — the server's replicate lack-of-fit verdict (each replicated design point
      against its OWN scatter, so it needs no sigma), read off ``residual_structure``. A flag stands
      at any df; a pass on few lack-of-fit df is recorded ``weak`` and still passes, as it does
      every other gate that reads this arm.
    * ``residual_structure`` — the server's sign (bias) and runs (trend) tests per observable,
      whose ``flagged_observables`` also carries any replicate lack-of-fit flag. Nothing flagged
      on underpowered tests (``low_power``) is no evidence either way.

    ``passed`` is ``None`` whenever the test could not speak, and the judge does NOT count that as
    a pass. A ``None`` report means UNREADABLE, never "no sigma estimated". ``basis`` is the test
    whose evidence was read (``None`` where the declared one is invalid for this fit or had nothing
    to read); ``reason`` is the one line the verdict text quotes.
    """
    out = {
        "test": test,
        "basis": None,
        "passed": None,
        "alpha": alpha,
        "chi2_gof_pvalue": chi2_gof_pvalue,
        "lof_status": None,
        "lof_weak": None,
        "lof_df": None,
        "lof_untested_status": None,
        "flagged_observables": None,
        "reason": None,
    }
    rolled = residual_structure if isinstance(residual_structure, dict) else {}
    if test == ADEQUACY_CHI2:
        if not isinstance(noise_sigma_report, dict):
            out["reason"] = (
                "no readable noise-sigma report, so whether the noise is fixed is unknown"
            )
        elif noise_sigma_report:
            out["reason"] = (
                f"the fit estimated noise sigma(s) {sorted(map(str, noise_sigma_report))}, which pin "
                "chi2 near n — the declared chi2 test is invalid for this fit"
            )
        else:
            out["basis"] = ADEQUACY_CHI2
            if chi2_gof_pvalue is None:
                out["reason"] = (
                    "no chi2 p-value (chi2 unreported, or no residual degrees of freedom)"
                )
            else:
                out["passed"] = chi2_gof_pvalue >= alpha
                out["reason"] = (
                    f"chi2 GoF p={chi2_gof_pvalue:.3g} {'>=' if out['passed'] else '<'} {alpha:g}"
                )
    elif test == ADEQUACY_REPLICATE_LOF:
        lof = rolled.get("replicate_lack_of_fit")
        if not isinstance(lof, dict):
            out["reason"] = "no replicate lack-of-fit verdict was read"
            return out
        status, lof_df = lof.get("status"), lof.get("lof_df")
        out.update(
            basis=ADEQUACY_REPLICATE_LOF,
            lof_status=status,
            lof_df=lof_df,
            # As in the history CSV (metrics._flatten_replicate_lof): only where a df was counted.
            lof_weak=None if lof_df is None else bool(lof.get("weak")),
            lof_untested_status=lof.get("untested_status"),
        )
        if status == "flagged":
            out["passed"] = False
            out["reason"] = (
                f"replicate lack-of-fit flags {lof.get('n_flagged', 0)} of {lof.get('n_points')} "
                "replicated design points"
            )
        elif status == "pass":
            out["passed"] = True
            out["reason"] = (
                f"replicate lack-of-fit passes on {lof.get('n_points')} design points"
                + (f" (weak: {lof_df} lack-of-fit df)" if out["lof_weak"] else "")
            )
            if out["lof_untested_status"]:
                out["reason"] += f"; a further family went untested ({out['lof_untested_status']})"
        else:
            out["reason"] = f"replicate lack-of-fit could not run ({status})"
    elif test == ADEQUACY_RESIDUAL_STRUCTURE:
        # `_residual_structure_fields` omits the verdict keys unless it READ them.
        if rolled.get("rollup_status") != "read":
            out["reason"] = "no residual-structure verdict was read"
            return out
        flagged = [str(o) for o in rolled.get("flagged_observables") or []]
        out.update(basis=ADEQUACY_RESIDUAL_STRUCTURE, flagged_observables=flagged)
        if flagged:
            out["passed"] = False
            out["reason"] = f"residual structure in {', '.join(flagged)}"
        elif rolled.get("low_power"):
            out["reason"] = "nothing flagged, but the sign/runs tests were underpowered"
        else:
            out["passed"] = True
            out["reason"] = "no residual bias or trend in any observable (tests powered)"
    else:
        out["reason"] = f"unknown adequacy test {test!r}"
    return out
