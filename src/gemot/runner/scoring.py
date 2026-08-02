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
    NOISE_SIGMA_REFERENCE_FACTOR as _NOISE_SIGMA_REFERENCE_FACTOR,
)

__all__ = [
    "_NOISE_SIGMA_COLLAPSE_FRACTION",
    "_NOISE_SIGMA_REFERENCE_FACTOR",
    "_noise_sigma_flag",
    "_reference_sigma_flag",
    "_residual_structure_fields",
]

def _residual_structure_fields(diagnosis) -> dict | None:
    """A compact, bounded residual-structure block for result.json from a get_fit_diagnosis
    payload, or ``None`` when no diagnosis is available (no simulation / model-less run).

    Keeps the rolled-up verdict (which observables show significant sign-test bias or runs-test
    time trend, and whether the tests were underpowered) plus one row per observable with the
    two p-values and the bias magnitude — enough to analyse a sweep without the per-point data.

    ``replicate_lack_of_fit`` is carried through verbatim: it is already the bounded projection the
    fit result rides on (a verdict plus at most REPLICATE_LOF_MAX_REPORTED_POINTS named points), and
    without it a sweep cannot tell whether the arm FLAGGED, PASSED (and on how many lack-of-fit
    degrees of freedom — ``weak``) or never ran. Without it a sweep cannot tell those apart
    from its result.json — establishing that the arm had fired at all meant going to the Phoenix
    trace, which does not scale to a series. ``metrics._flatten_replicate_lof`` takes it the last
    step, into the flat ``lof_*`` columns of the history CSV a sweep is actually grouped by.
    """
    if not isinstance(diagnosis, dict):
        return None
    rolled = diagnosis.get("residual_structure")
    per = diagnosis.get("per_observable")
    if not isinstance(rolled, dict) and not isinstance(per, list):
        return None
    # NARROWED once, here, not `(rolled or {})` at each use: the guard above is satisfied by `per`
    # alone, so a truthy NON-dict `residual_structure` (a status string from a degraded payload) got
    # past it and `.get` raised AttributeError — which judge.py catches by setting the whole block to
    # None, losing the per-observable rows too. This is the projection that makes a run readable from
    # result.json; it must degrade, not vanish.
    rolled = rolled if isinstance(rolled, dict) else {}
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
        "flagged_observables": rolled.get("flagged_observables", []),
        "low_power": bool(rolled.get("low_power")),
        "per_observable": per_obs,
    }
    lof = rolled.get("replicate_lack_of_fit")
    if isinstance(lof, dict):
        out["replicate_lack_of_fit"] = lof
    return out


def _reference_sigma_flag(report, reference) -> dict | None:
    """Plausibility of the candidate's ESTIMATED sigmas against the TRUE model's OWN fitted sigmas
    (``reference.json``'s ``noise_sigma``), or ``None`` when the reference doesn't record them.

    This is the dimensionally sound comparison and the judge's preferred one (#118): the marginal
    data std conflates variance the model EXPLAINS with variance it cannot, which is why legitimate
    sigmas span ~0.0004-0.59 of it and no fixed band can hold the truth. Against the reference's own
    sigma the scale question disappears — a correct fit lands near 1 — so the only choice left is a
    tolerance FACTOR, kept generous because a good model that isn't literally the truth still has
    somewhat different residuals (a sharp chi2 point-null rejects near-misses once n is large).

    Returns ``{inflated, collapsed, max_ratio, min_ratio}`` where the ratios are
    ``fitted / reference`` per sigma.
    """
    if not isinstance(report, dict) or not report:
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
    hi = lo = None
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
    if hi is None:
        return None
    return {"inflated": hi > _NOISE_SIGMA_REFERENCE_FACTOR,
            "collapsed": lo < 1.0 / _NOISE_SIGMA_REFERENCE_FACTOR,
            "max_ratio": hi, "min_ratio": lo,
            # Same key names the data-std fallback emits, so a consumer of result.json's
            # noise_sigma_flag sees ONE shape regardless of which basis produced it.
            "max_fraction_of_data_std": _max_fraction_of_data_std(report),
            "min_fraction_of_data_std": _min_fraction_of_data_std(report)}


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
