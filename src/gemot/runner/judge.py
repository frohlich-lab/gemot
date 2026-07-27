"""The independent verdict: score the built model against the true model by BIC.

Deliberately structure-agnostic — it does not check whether the agent found "the" mechanism, only
whether what it built explains the data at a competitive parameter count (``delta_bic <=
scenario.bic_threshold``), with the noise-sigma plausibility gate as the second condition so a
good BIC bought with a gamed noise model does not pass. The scenario's own ``diagnostics`` run
report-only alongside.

Scores the run's BEST checkpoint, not whatever HEAD happens to be: a run that ended on a worse
model is rolled back first.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .clock import _ToolClock
from .provenance import _bic_reference, _now_iso
from ..scenario import Baseline, JudgeContext, Scenario
from .scoring import _noise_sigma_flag, _reference_sigma_flag, _residual_structure_fields


async def _head_version(call) -> "int | None":
    """The current HEAD version number, read off ``describe_problem``.

    There is no separate ``current_version`` tool: everything it returned (``current_version``,
    ``has_unfit_edits``, ``n_pending_edits``) is already on ``describe_problem``, so the second
    tool was a schema in every client's context for no extra capability. ``None`` when the read
    fails, so a caller can fall back rather than abort.
    """
    try:
        desc = await call("describe_problem")
    except Exception:  # pragma: no cover - defensive; callers degrade
        return None
    return int(desc.get("current_version", 0)) if isinstance(desc, dict) else None


async def _restore_best_checkpoint(call) -> None:
    """Make the early-stop chain's best-BIC model the current HEAD before scoring.

    Agents wander off their best model and finish worse (verified pass→fail flips on a credit
    abort, e.g. armistead v56 → v66), so the judge must score the BEST fitted version, not the
    last-committed one. The server tracks that checkpoint (``get_progress`` → ``best_version``);
    ``restore_version`` restores it as a new HEAD, which every read in :func:`_judge` then sees. This
    runs on ANY termination that reaches the judge (early-stop, wall-time, recursion limit,
    clean finish). Best-effort — a probe/rollback failure just leaves HEAD in place.

    Edits are fit-free (issue #81), so an agent can assemble a model yet never ``fit`` it — leaving
    no checkpoint and an unfit working model. The judge must never score an edited-but-unfit model,
    so when there is no checkpoint but a model exists, fit it once here before scoring.
    """
    try:
        progress = await call("get_progress")
    except Exception:  # pragma: no cover - older server without the tool: score HEAD
        return
    best = progress.get("best_version") if isinstance(progress, dict) else None
    if best is None:
        # No fitted checkpoint yet. If a model was assembled but never fit, score it fitted:
        # fit once now (issue #81's "auto-fit on finish"), then adopt the resulting checkpoint.
        try:
            desc = await call("describe_problem")
        except Exception:  # pragma: no cover - defensive
            desc = {}
        if isinstance(desc, dict) and desc.get("model_initialized"):
            try:
                await call("fit", justification="score the assembled model before judging")
                progress = await call("get_progress")
                best = progress.get("best_version") if isinstance(progress, dict) else None
                if best is not None:
                    print(f"[judge] fit the unscored working model -> checkpoint v{best} before scoring")
            except Exception as exc:  # pragma: no cover - fall through to scoring HEAD
                print(f"[judge] could not fit the working model before scoring "
                      f"({type(exc).__name__}: {exc})")
        if best is None:
            return
    head = await _head_version(call)
    if best == head:
        return  # already on the best model
    try:
        await call("restore_version", version=best,
                   justification="restore best-BIC checkpoint before scoring")
        print(f"[judge] restored best-BIC checkpoint v{best} (was HEAD v{head}) before scoring")
    except Exception as exc:  # pragma: no cover - score HEAD if the restore fails
        print(f"[judge] could not restore best-BIC checkpoint v{best} "
              f"({type(exc).__name__}: {exc}); scoring HEAD instead")


async def _judge(scenario: Scenario, call, baseline: Baseline, work: Path, meta: dict,
                 clock: "_ToolClock", t0: float) -> "tuple[int, dict]":
    """Judge the repaired model against the true model by BIC (structure-agnostic).

    Success ⇔ the repaired model can't be rejected relative to the true (data-generating)
    model — ``BIC(candidate) - BIC(true) <= scenario.bic_threshold`` — AND its estimated noise
    sigma is plausible (not inflated to absorb misfit or collapsed to overfit the noise; see
    ``_noise_sigma_flag``). The sigma check is a HIDDEN integrity gate: it never reaches the agent,
    so a good BIC bought with an implausible noise model can't slip through and can't be gamed
    around. We don't require any particular rate-law FORM (models need not be structurally
    identifiable) and we don't special-case invented species — BIC's ``k·ln(n)`` penalty makes an
    extra parameter earn its keep. A χ² goodness-of-fit p-value (residuals vs the known noise) is
    reported alongside. The scenario's optional ``diagnostics`` hook adds report-only detail.

    Returns ``(exit_code, result)``: the process exit code (0 on success, 1 otherwise) and
    the machine-readable result dict (also written to ``work/result.json``), so the caller
    can stamp the outcome onto the run span.
    """
    from ..scenario import bic, chi2_gof_pvalue

    # Restore + score the best-BIC checkpoint, not whatever was committed last (see the helper).
    await _restore_best_checkpoint(call)

    structure = await call("get_model_structure")
    reactions = structure.get("reactions", []) or []
    current_species = set(structure.get("species", []) or [])
    new_species = sorted(current_species - baseline.species)

    final = await call("get_fit_summary")
    # A degenerate or never-built model (e.g. the agent authored nothing, or a fit that produced
    # a nan/inf objective) has no usable neg_log_likelihood — get_fit_summary returns None, or a
    # non-finite value. Treat any of those as "no valid fit" (final_nll = None) and score the run
    # as a clean failure below, rather than crashing the judge on float(None) / a non-finite BIC.
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
    # k. Both sides must count free parameters the same way.
    params = await call("get_parameters")
    k = sum(1 for p in params if p.get("estimate")) if isinstance(params, list) else 0
    chi2 = final.get("chi2") if isinstance(final, dict) else None
    chi2 = float(chi2) if chi2 is not None else None
    # Residual-structure tests (sign / runs) of the SCORED model — the checkpoint was already
    # restored above, so this is the fit that gets judged. Recorded for post-hoc analysis: e.g.
    # whether runs that passed the BIC gate also had structure-free residuals, and whether the
    # tests were underpowered. Report-only: a diagnosis failure must never fail the run (which
    # would leave no result.json at all — the exact unscored-run outcome the harness avoids), so
    # the whole call is guarded and degrades to None.
    try:
        residual_structure = _residual_structure_fields(await call("get_fit_diagnosis"))
    except Exception as exc:  # noqa: BLE001 - report-only diagnostic; never fail the judge on it
        print(f"  (residual-structure diagnosis unavailable: {type(exc).__name__}: {exc})")
        residual_structure = None
    head = await _head_version(call) or 0
    diff = await call("diff_versions", from_version=0, to_version=head) if head else {}
    errors = await call("get_errors", limit=20)
    # baseline.nll is None for a zero-shot build (no model shipped → nothing to score).
    delta = (baseline.nll - final_nll) if (baseline.nll is not None and final_nll is not None) else None

    # Estimated-noise-sigma plausibility of the SCORED model. A wrong model can reach a good BIC
    # by making the noise model implausible — inflating sigma to absorb misfit, or collapsing it to
    # overfit the noise — so an implausible sigma disqualifies an otherwise-passing BIC. This is a
    # HIDDEN integrity gate: applied post-run in the judge and NEVER surfaced to the agent (which
    # only ever sees its own noise_sigma_report, not this verdict), so it can't be gamed around.
    # The raw report is stored in result.json too (it wasn't before) to audit gaming after the fact.
    noise_sigma_report = final.get("noise_sigma_report") if isinstance(final, dict) else None
    ref = scenario.reference()
    # PREFER the reference-sigma comparison (#118): a candidate sigma is judged against the TRUE
    # model's own fitted sigma, not against a fraction of the marginal data std (which conflates
    # explained and unexplained variance, so no fixed band holds the truth). Falls back to the
    # data-std band for a reference.json written before `noise_sigma` was recorded — regenerate
    # with `python examples/setup.py --refit <example>` to switch an example over.
    reference_sigma_flag = _reference_sigma_flag(noise_sigma_report, ref)
    noise_sigma_flag = reference_sigma_flag or _noise_sigma_flag(noise_sigma_report)
    sigma_basis = "reference" if reference_sigma_flag else "data_std"
    sigma_implausible = bool(noise_sigma_flag
                             and (noise_sigma_flag["inflated"] or noise_sigma_flag["collapsed"]))

    # BIC gate against the true-model reference, AND the hidden sigma-plausibility gate.
    bic_cand = bic(final_nll, k, n) if (n and final_nll is not None) else None
    bic_ref = _bic_reference(ref)
    delta_bic = (bic_cand - bic_ref) if (bic_cand is not None and bic_ref is not None) else None
    bic_pass = delta_bic is not None and delta_bic <= scenario.bic_threshold
    success = bic_pass and not sigma_implausible
    # χ² goodness-of-fit vs the known noise (reported, not gating).
    dof = max(0, n - k)
    p_gof = chi2_gof_pvalue(chi2, dof)

    ctx = JudgeContext(call=call, baseline=baseline, final_nll=final_nll, structure=structure,
                       new_species=new_species, delta=delta, n=n, k=k, chi2=chi2)
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
    print(f"  fit: nll={_nll}  chi2={chi2:.6g}  k={k}  n={n}  (baseline nll {_bl}, Δnll {_dl})"
          if chi2 is not None else
          f"  fit: nll={_nll}  k={k}  n={n}  (baseline nll {_bl}, Δnll {_dl})")
    if delta_bic is not None:
        print(f"  BIC:  candidate {bic_cand:.6g} | true-model {bic_ref:.6g} | "
              f"ΔBIC {delta_bic:+.6g}  (accept if <= {scenario.bic_threshold:g})")
    else:
        # A reference that parses but lacks nll/k/n makes bic_ref None while `ref` is a truthy
        # dict — report that as "malformed", not "ok", so the real cause of the missing gate shows.
        ref_state = "missing" if not ref else ("ok" if bic_ref is not None else "malformed")
        print(f"  BIC:  unavailable (reference {ref_state}, n={n})")
    if p_gof is not None:
        print(f"  chi2 GoF:             p={p_gof:.3g} (dof={dof}) "
              f"— {'consistent with noise' if p_gof >= 0.05 else 'worse than noise'}")
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
            breakdown = (f" [compile {compile_s:.0f}s ×{n_compiles} + fit {fit_s:.0f}s ×{n_fits}"
                         f" + other {max(0.0, compute_s - compile_s - fit_s):.0f}s]")
        print(f"  time:                 total {wall:.0f}s | tool calls {compute_s:.0f}s "
              f"({clock.calls} calls){breakdown} | LLM+overhead {llm_s:.0f}s")
    if errors:
        print(f"  rejected tool calls:  {len(errors)} (see get_errors)")
    ant_diff = diff.get("model_diff") if isinstance(diff, dict) else None
    if ant_diff:
        print("  model diff (v0 -> HEAD):")
        for line in str(ant_diff).splitlines():
            print("    " + line)
    if success:
        verdict = (f"not rejectable vs the true model (ΔBIC {delta_bic:+.3g} <= "
                   f"{scenario.bic_threshold:g}); the repair explains the data as well as the truth.")
    elif final_nll is None:
        verdict = "no valid fit (the model was never built or produced a non-finite objective)."
    elif delta_bic is None:
        verdict = "could not evaluate BIC (missing reference.json or no data points)."
    elif bic_pass and sigma_implausible:
        which = " and ".join(
            w for w, on in (("inflated (absorbing misfit)", noise_sigma_flag["inflated"]),
                            ("collapsed (overfitting the noise)", noise_sigma_flag["collapsed"]))
            if on)
        verdict = (f"BIC passes (ΔBIC {delta_bic:+.3g}) but the estimated noise sigma is {which} — "
                   "the good score rests on an implausible noise model, so it does not count.")
    else:
        verdict = (f"rejectable: ΔBIC {delta_bic:+.3g} > {scenario.bic_threshold:g} — the fit is "
                   "decisively worse than the true model.")
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
        "n": n,
        "bic": bic_cand,
        "bic_reference": bic_ref,
        "delta_bic": delta_bic,
        "bic_threshold": scenario.bic_threshold,
        "chi2_dof": dof,
        "chi2_gof_pvalue": p_gof,
        "residual_structure": residual_structure,
        # Noise-sigma plausibility of the scored model: GATES success (an inflated/collapsed sigma
        # disqualifies an otherwise-passing BIC; see the gate above) but is never shown to the agent.
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
        "run_ended_at": _now_iso(),
        **diag_fields,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    print(f"  (machine-readable result -> {(work / 'result.json').relative_to(scenario.example_dir.parent)})")
    return (0 if success else 1), result
