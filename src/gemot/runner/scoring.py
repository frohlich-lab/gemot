"""Projections of the server's fit reports into the judge's verdict fields.

Two families, both pure functions over a report payload:

* the compact residual-structure block ``result.json`` carries, and
* the noise-sigma plausibility gates — is the estimated sigma inflated to absorb misfit, or
  collapsed to overfit the noise? The preferred gate compares against the TRUE model's own
  fitted sigmas (dimensionally sound); the data-std band is the fallback where a reference
  doesn't record them.

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

__all__ = [
    "_NOISE_SIGMA_COLLAPSE_FRACTION",
    "_NOISE_SIGMA_BAND_ALPHA",
    "_NOISE_SIGMA_BAND_MIN_DF",
    "_NOISE_SIGMA_REFERENCE_FACTOR",
    "_noise_sigma_flag",
    "_reference_sigma_flag",
    "_residual_structure_fields",
]

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
    for o in (per or []):
        if not isinstance(o, dict):
            continue
        st = o.get("sign_test") or {}
        rt = o.get("runs_test") or {}
        per_obs.append({
            "observable_id": o.get("observable_id"),
            "sign_test_p": st.get("p"),
            "runs_test_p": rt.get("p"),
            "mean_weighted_residual": o.get("mean_weighted_residual"),
            "low_power": bool(o.get("structure_low_power")),
        })
    out = {
        "rollup_status": "read" if read else "unavailable",
        "rows_status": "read" if isinstance(per, list) else "unavailable",
        "per_observable": per_obs,
    }
    if read:
        out["flagged_observables"] = rollup.get("flagged_observables", [])
        out["low_power"] = bool(rollup.get("low_power"))
        lof = rollup.get("replicate_lack_of_fit")
        if isinstance(lof, dict):
            out["replicate_lack_of_fit"] = lof
    return out


def _reference_sigma_flag(report, reference, k=None, n=None) -> dict | None:
    """Plausibility of the candidate's ESTIMATED sigmas against the TRUE model's OWN fitted sigmas
    (``reference.json``'s ``noise_sigma``), or ``None`` when the reference doesn't record them.

    This is the dimensionally sound comparison and the judge's preferred one (#118): the marginal
    data std conflates variance the model EXPLAINS with variance it cannot, which is why legitimate
    sigmas span ~0.0004-0.59 of it and no fixed band can hold the truth. Against the reference's own
    sigma the scale question disappears — a correct fit lands near 1.

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
    ``fitted / reference`` per sigma and ``per_sigma`` carries each one's ratio, df and band.
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
    # model. Until then, decline the whole basis as the unpairable-sigma case below does, so `_judge`
    # falls through to the data-driven band -- the test that was right all along here: it never
    # flagged any of the 116 runs scored on it, while this basis silently failed 17 fiedler runs that
    # had BEATEN the BIC bar. See reconcile.sigma_scale_coupled.
    if any(e.get("scale_coupled") for e in report.values() if isinstance(e, dict)):
        return None
    # Prefer the EFFECTIVE sd on BOTH sides: for a composite/proportional noiseFormula the raw
    # parameter is not comparable across models (fiedler's sd is per-gel scale * sigma, so a
    # candidate at a different amplitude trades scale against sigma and reaches an IDENTICAL sd
    # with a raw sigma several-fold off — judged raw, it fails for matching the truth). Fall back
    # to raw-vs-raw for a reference written before noise_sigma_effective existed.
    reference = reference if isinstance(reference, dict) else {}
    effective_ref = reference.get("noise_sigma_effective")
    use_effective = isinstance(effective_ref, dict) and bool(effective_ref)
    ref_sigmas = effective_ref if use_effective else reference.get("noise_sigma")
    if not isinstance(ref_sigmas, dict) or not ref_sigmas:
        return None
    # The reference is fit on the PRISTINE collection problem, whose parameter ids can differ from
    # the materialized ones an agent sees (perelson ships `sd_viral_load` but the collection calls
    # it `sd_task0_model0_perelson1_V`). When BOTH sides have exactly one sigma there is no
    # ambiguity about the pairing, so match them positionally rather than losing the comparison —
    # anything more than one-to-one stays id-matched, since a wrong pairing is worse than none.
    solo_pairing = len(report) == 1 and len(ref_sigmas) == 1
    solo_ref_value = next(iter(ref_sigmas.values())) if solo_pairing else None
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
        # Match the side chosen above: effective vs effective, or raw vs raw.
        fitted = (entry.get("effective_sigma") if use_effective and "effective_sigma" in entry
                  else entry.get("fitted"))
        ref_value = ref_sigmas.get(pid, solo_ref_value) if solo_pairing else ref_sigmas.get(pid)
        try:
            fitted, ref_value = float(fitted), float(ref_value)
        except (TypeError, ValueError):
            # EVERY reported sigma must be pairable and usable. Skipping one and judging on the
            # rest would let an agent rename (or zero) the sigmas it games and keep one honest
            # sigma matching, so the gate reports max_ratio ~ 1 and passes a model whose noise was
            # gamed on every other observable. Give up instead, so `_judge` falls through to the
            # data-std band rather than accepting a partial verdict.
            return None
        if not (fitted > 0 and ref_value > 0):
            return None
        ratio = fitted / ref_value
        hi = ratio if hi is None else max(hi, ratio)
        lo = ratio if lo is None else min(lo, ratio)
        floor, ceiling, df_cand, df_ref = _sigma_ratio_band(
            entry.get("n_points"), k, n, reference, alpha)
        detail = {"ratio": ratio, "n_points": entry.get("n_points"),
                  "band_low": floor, "band_high": ceiling,
                  "df": df_cand, "df_reference": df_ref}
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
    return {"inflated": inflated, "collapsed": collapsed,
            "max_ratio": hi, "min_ratio": lo,
            # The per-sigma detail behind the verdict: each ratio, the df it was judged at, and the
            # band that df implies. Written to result.json so a flag (or the absence of one on a
            # low-power sigma) can be audited without re-deriving any of it.
            "alpha": alpha, "per_sigma": per_sigma,
            # Same key names the data-std fallback emits, so a consumer of result.json's
            # noise_sigma_flag sees ONE shape regardless of which basis produced it.
            "max_fraction_of_data_std": _max_fraction_of_data_std(report),
            "min_fraction_of_data_std": _min_fraction_of_data_std(report)}


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

    Tolerance is MONOTONE in df: fewer points can only widen the band, never narrow it. That sounds
    obvious and an earlier draft got it backwards — below ``NOISE_SIGMA_BAND_MIN_DF`` it switched to
    the fixed factor, so at brannmark's shape a FIVE-point sigma was judged in [0.333, 3.0] while a
    seven-point one got [0.154, 8.65]: the thinner evidence, the harsher test, and the very
    false-failure this change exists to remove would have survived just below the floor. The floor
    now clamps the df the band is COMPUTED at rather than switching regimes, so tolerance stops
    widening below it instead of collapsing. A missing point count is the one case with no df at
    all, and falls back to the fixed factor.
    """
    from scipy.stats import f as _f

    fixed = (1.0 / _NOISE_SIGMA_REFERENCE_FACTOR, float(_NOISE_SIGMA_REFERENCE_FACTOR))
    try:
        points = int(n_points)
        k_cand, n_cand = int(k), int(n)
        k_ref, n_ref = int(reference["k"]), int(reference["n"])
    except (TypeError, ValueError, KeyError):
        return (*fixed, None, None)       # unknown df: the fixed tolerance, as before
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
    return (min(fixed[0], float(_f.ppf(alpha / 2, lo_df, hi_df)) ** 0.5),
            max(fixed[1], float(_f.ppf(1 - alpha / 2, lo_df, hi_df)) ** 0.5),
            df_cand, df_ref)


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
    """Report-only integrity marker: is any ESTIMATED noise sigma implausible relative to the data
    — INFLATED past its data-driven cap (absorbing model misfit) or COLLAPSED far below the data
    std (overfitting the noise for a suspiciously good BIC)? Returns
    ``{inflated, collapsed, max_fraction_of_data_std, min_fraction_of_data_std}`` or ``None`` when
    the problem has no estimated sigma.

    An inflated/collapsed sigma DISQUALIFIES an otherwise-passing BIC (see _judge) — a hidden
    integrity gate applied post-run and never surfaced to the agent. The full block is also written
    to result.json so a gated (or borderline) run is auditable. The collapse threshold
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
    return {"inflated": inflated, "collapsed": collapsed,
            "max_fraction_of_data_std": hi, "min_fraction_of_data_std": lo}
