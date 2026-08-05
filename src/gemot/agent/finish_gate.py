"""``finish`` — the lead-only control tool, and the only legitimate way to end a run.

Plain iteration needs no call here; the tool exists to GATE a finish. A finish must rest on a
current, fitted model with a plausible noise sigma, and takes a two-call ARM→CONFIRM handshake so
a diagnosis is always shown before a run is allowed to end. A silent stop is bounced back into
this gate by the driver, so it cannot be sidestepped.

Every string this returns to the model lives in :mod:`petab_harness.prompts`; the counters and
the relent rules are here.
"""

from __future__ import annotations

import math

from .. import prompts
from .probe import _current_diagnosis, _current_fit

# After this many CONSECUTIVE finish attempts with no valid fit, the gate gives up: it flags the
# run (finish_state["gave_up"]) so the driver ends it promptly as no-model-built, rather than
# refusing forever while the model spams finish to the recursion limit. It is NOT an accepted
# finish — there is no fitted model to score — just a bounded, honest termination.
_MAX_NO_FIT_FINISH_ATTEMPTS = 5

# A finish on a model whose ESTIMATED noise sigma is implausible (inflated to absorb misfit, or
# collapsed to overfit the noise — the server flags this in get_fit_summary) is REFUSED with the
# fix-the-mechanism advice, so the agent doesn't submit a BIC that rests on a gamed noise model
# (the judge fails those anyway). Bounded like the no-fit gate: after this many consecutive refusals
# it relents (lets the finish through) so a model that genuinely can't reach a plausible sigma still
# terminates cleanly at its honest — losing — score, instead of spamming finish to the step limit.
_MAX_SIGMA_FINISH_REFUSALS = 3

# A finish on an OVER-parameterised model — one whose bic rests on a parameter combination the data
# cannot constrain, so the FIM at the optimum is RANK-DEFICIENT (rank < n_estimated) — is REFUSED
# with GENERAL identifiability advice, so the agent doesn't submit a good-looking bic inflated by
# free parameters the experiment can't pin (the judge fails those, and bic should have penalised
# them). This is the sigma gate's structural blind spot: extra free parameters keep the per-point
# fit — and thus the ESTIMATED sigma — tight, so the sigma gate passes while k is padded (the
# issue-driving case: a steady-state fit that freely estimated all 16 initial conditions,
# non-identifiable because equilibrium is IC-independent, finished at a great BIC with the sigma
# gate none the wiser — and a rank-deficient FIM). The gate keys on rank deficiency ALONE, NOT on
# the mere presence of a near-flat eigen-direction: calibration across the reference structures
# showed the latter over-refuses legitimate, identifiable fits (a parameter at a bound, or a
# correlated-but-full-rank group, routinely produces a flat_direction while the FIM stays full
# rank). Bounded exactly like the sigma gate: after this many consecutive refusals it RELENTS, so a
# model that genuinely cannot be made identifiable still terminates at its honest score instead of
# spamming finish. An identifiable (full-rank) fit RESETS the counter. FAILS OPEN — an unavailable or
# STALE diagnosis (no FIM, a diagnosis of a DIFFERENT structure, or the probe returned {}) never
# refuses a finish. The advice is deliberately GENERAL (name the flat-direction params, point at the
# simplification menu); the specific fix is the agent's to reason out, not the gate's to hand over.
_MAX_PARSIMONY_FINISH_REFUSALS = 3

# A finish on a model the REPLICATE LACK-OF-FIT arm condemns is REFUSED. This carries what the
# sigma-ratio arm used to: a fitted sigma several-fold above the pure-error floor was an averaged,
# thresholded proxy for "the predictions miss design points by more than their own replicate scatter
# allows", which this arm measures directly and per point. Without it here that whole class of run
# loses its refusal — the arm's verdict would only be DISPLAYED at arm time, which a second finish
# call clears (blasi's dBIC +437 run: Holm p 1.5e-18, sigma under the data-std cap, FIM full rank at
# k=6, so neither of the gates above sees it).
#
# The replicate arm ONLY. The sign/runs tests share `flagged_observables` with it but are advisory by
# design (residual_tests: low power at these sample sizes, so a flag is weak evidence), and gating on
# them would refuse far beyond what this replaces. What this gate reads is the arm's FLAG, which is
# the part with power at any lack-of-fit df: Holm-adjusted at REPLICATE_LOF_ALPHA against each point's
# own replicate scatter, and reported as NO EVIDENCE (available False) rather than as a flag wherever
# the floor is thin, the family too small, or the pooled df <= 0 — so it FAILS OPEN on its own. Its
# PASS is the qualified half (`weak`, `lof_df`: constants.REPLICATE_LOF_MIN_LOF_DF), and nothing here
# reads that — a weak pass clears this gate exactly as a powered one does, deliberately; it reaches
# the lead as prose, in the armed message's `residual_structure.summary`. Bounded and reset exactly
# like the two gates above.
_MAX_STRUCTURE_FINISH_REFUSALS = 3


def _render_lof_points(points, *, max_points: int = 4) -> str:
    """The condemned design points rendered for the refusal text, as ``label (z, Holm p)`` — the
    server already caps this list (REPLICATE_LOF_MAX_REPORTED_POINTS) and orders it worst-first, so
    the agent sees WHERE the form fails, not just that it does."""
    named = [p for p in (points or []) if isinstance(p, dict)][:max_points]
    if not named:
        return "(see get_fit_diagnosis for the condemned points)"
    return ", ".join(f"{p.get('label')} (z={p.get('z')}, Holm p={p.get('p_holm')})" for p in named)


def _render_flat_directions(flat_directions, *, max_dirs: int = 4, max_parts: int = 4) -> str:
    """The FIM's near-flat (non-identifiable) eigen-directions rendered for the refusal text — as
    `id·weight` groups, one bracket per direction, so the agent sees WHICH parameters trade off
    TOGETHER (the collinear structure), not just a flattened list of names. A direction dominated
    by a single high-weight id is one unconstrained parameter; a direction spreading weight across
    several is a collinear group to merge. Bounded: at most ``max_dirs`` directions (the smallest
    eigenvalues, which the diagnosis already sorts first) and ``max_parts`` participants each,
    with a trailing marker when more were dropped — a wholesale-unidentifiable model (every initial
    condition free) shows a handful rather than a wall."""
    groups: list[str] = []
    for d in (flat_directions or [])[:max_dirs]:
        terms = [f"{p.get('id')}·{p.get('weight')}"
                 for p in (d.get("participants") or [])[:max_parts] if p.get("id")]
        if terms:
            groups.append("[" + ", ".join(terms) + "]")
    if not groups:
        # rank_deficient can be flagged with no bounded flat-direction participants to name. Names
        # the SECTION, not the bare tool: `sections` defaults to the residual half, which carries no
        # flat directions at all, so a lead following this would spend the turn on the wrong payload.
        return "(see get_fit_diagnosis(sections='parameters') for the flat directions)"
    return "; ".join(groups) + ("; …" if len(flat_directions) > max_dirs else "")


def _render_merge(hints) -> str:
    """The OUTSTANDING MERGES line for the ARM text, or ``""`` when there is no tie to report.

    Read off ``get_fit_summary``, which now carries the same ``merge_hints`` the fit result does, so
    the gate names the pairs the lead was already shown. Fails OPEN like every other diagnosis read
    here: a missing field or an odd shape means "no merge to report", never a raised finish — and one
    unreadable row does not silence the readable ones.

    ``dbic`` is NEGATIVE when the merge wins and the template says "LOWERS bic by", so the slot takes
    the MAGNITUDE — through :func:`_dbic_magnitude`, because ``abs(hint["dbic"] or 0.0)`` rendered
    "LOWERS bic by 0.0" for a tie that beats the price by under 5e-4 nll (its dbic rounds to -0.0,
    which is FALSY) while telling the lead the model was not finished. An edit whose stated payoff is
    zero is not an argument; the same ``or`` swallowed a genuinely absent dbic into the same 0.0.
    """
    if not isinstance(hints, (list, tuple)):
        return ""
    ties: list[str] = []
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        try:
            params = [str(p) for p in (hint.get("parameters") or ())]
            if not params:
                continue
            ties.append(prompts.FINISH_ARMED_MERGE_TIE_TMPL.format(
                parameters=" == ".join(params), dnll=hint.get("dnll"),
                dbic=_dbic_magnitude(hint.get("dbic")), move=hint.get("move_scaled")))
        except Exception:  # noqa: BLE001 - an advisory line never fails a finish
            continue
    if not ties:
        return ""
    try:
        return prompts.FINISH_ARMED_MERGE_TMPL.format(ties="".join(ties))
    except Exception:  # noqa: BLE001 - inside the fail-open boundary too: a template slot
        return ""      # mistake must cost the advisory line, never every finish in the run


def _dbic_magnitude(dbic) -> str:
    """How much bic the merge LOWERS, as displayed: the magnitude, ``unknown`` when there is none,
    and a BOUND rather than ``0.0`` when the win is real but below the reported resolution (dbic is
    ``2*(dnll - price)`` rounded to 3 dp, and the hint only exists where ``dnll < price``)."""
    if not isinstance(dbic, (int, float)) or isinstance(dbic, bool):
        return "unknown"
    magnitude = abs(float(dbic))
    return f"{magnitude:g}" if round(magnitude, 3) else "less than 0.001"


def _finish_tool(session, finish_state=None, probe=None):
    """The lead-only control tool for the one non-default move: ending the run.

    ``probe`` is the driver's UN-TIMED ``await probe(name, args)`` handle; the gate's two internal
    polls — get_fit_summary for the score, get_fit_diagnosis for the residual detail and the FIM —
    go through it so they don't inflate the run's tool_calls / compute_seconds (falling back to the
    timed ``session`` handle when absent).

    Plain iteration needs no call — to keep going the lead simply makes the next ``make_changes``
    edit — so the tool exists only to gate a finish.

    ``finish`` is the ONLY legitimate way to end a run: a run that silently stops (a no-tool
    turn) is bounced back into this gate by the driver (see run_deepagents), so the gate cannot
    be sidestepped. To finish, the model MUST have a fitted model — a finish with no valid fit,
    or with edits made since the last fit, is REFUSED and never relented (the harness does not
    accept an unfitted model as finished; the run ends by other means — the step/wall limits —
    rather than by accepting an unscored result). A valid finish is a TWO-call handshake: the
    first call ARMS and presents the fit diagnosis (bic + residual structure), the second
    CONFIRMS — so a diagnosis is always shown before a run is allowed to end. ``finish_state``
    (a shared dict) records ``confirmed`` for the driver, so a properly-confirmed finish is not
    bounced.

    It deliberately does NOT gate on the χ² goodness-of-fit (chi2_gof_pvalue / reduced_chi2).
    Whenever the observable noise sigmas are estimated (as in the issue-#54 benchmarks — though
    not every problem frees them), χ² pins near the data-point count by construction and its
    p-value is uninformative, so gating ``finish`` on p ≥ 0.05 made the panel abandon its
    best-BIC model to chase an unreachable target (issue #54). Where χ² cannot be computed at
    all, the tool SAYS so rather than surfacing it as a fit rejection.
    """
    from langchain_core.tools import StructuredTool

    finish_state = finish_state if finish_state is not None else {}
    finish_state.setdefault("confirmed", False)
    finish_state.setdefault("gave_up", False)
    # Read HEAD's fit/diagnosis through the un-timed probe when the driver wires one, so the
    # gate's own polling doesn't count toward the tool clock / tool_calls metric; fall back to
    # the (timed) session handle otherwise.
    reader = probe if probe is not None else session.call_tool

    # `armed_token` identifies the fit a finish was ARMED on (the diagnosis has been shown and a
    # confirming second finish is awaited), or None when not armed. The token is the fit's
    # identity, so an edit + refit between the two calls re-arms (the shown diagnosis would
    # otherwise be stale). It is NOT the get_fit_summary "version" alone: a restored payload
    # carries none (the persisted per-version sim blob has no version), and `None != None` would
    # then skip arming and confirm on the first call — bypassing the gate. bic+nll identify the
    # scored fit whether or not a version is attached.
    state = {"armed_token": None, "no_fit_refusals": 0, "sigma_refusals": 0,
             "parsimony_refusals": 0, "structure_refusals": 0}

    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    # `rationale` is REQUIRED (no default): ending the run must cost a written justification, the
    # deliberate argument the old `action` string used to be. An argless call is a schema error the
    # model is handed back, not a first half of the arm->confirm handshake.
    async def finish(rationale: str) -> str:
        print(f"[lead finish] {rationale}")
        fit = await _current_fit(reader)
        nll = fit.get("neg_log_likelihood")
        bic = fit.get("bic")
        version = fit.get("version")
        # Require a finite BIC — the score the harness actually finishes on. A finite nll is
        # necessary but not sufficient: get_fit_summary reports bic=None when n_data_points/k
        # are incomplete, and finishing then would report "bic=n/a" while the refusal text
        # promises "there is no bic to report". Gate on the score that must exist.
        has_fit = _finite(nll) and _finite(bic)
        # Block finish when there is no valid fit to finish on (nothing built, a non-finite
        # objective, or no computable bic) — a genuine failure, not a goodness-of-fit
        # judgement. We do NOT gate on chi2_gof_pvalue/reduced_chi2 (see the docstring): they
        # are uninformative when the noise sigmas are estimated, and chasing p ≥ 0.05 costs
        # the best-BIC model.
        if not has_fit:
            # NEVER relented into an accepted finish: the harness does not record an unfitted
            # model as finished (there is no bic to score). But it does not refuse FOREVER
            # either — after _MAX_NO_FIT_FINISH_ATTEMPTS consecutive no-fit attempts it GIVES
            # UP: flags the run so the driver ends it promptly as no-model-built, instead of
            # letting the model spam finish to the recursion limit.
            state["armed_token"] = None
            state["no_fit_refusals"] += 1
            if state["no_fit_refusals"] >= _MAX_NO_FIT_FINISH_ATTEMPTS:
                finish_state["gave_up"] = True
                print(f"[finish GAVE UP] no fittable model after {state['no_fit_refusals']} "
                      "attempts — ending the run as no-model-built")
                return prompts.FINISH_GAVE_UP_NO_FIT_TMPL.format(
                    attempts=state["no_fit_refusals"])
            print(f"[finish BLOCKED {state['no_fit_refusals']}/{_MAX_NO_FIT_FINISH_ATTEMPTS}] "
                  "no valid fit (no model built, non-finite objective, or no bic)")
            return prompts.FINISH_REFUSED_NO_FIT
        # A valid fit exists -> reset the no-fit give-up counter.
        state["no_fit_refusals"] = 0
        # There used to be a third gate here, for a model edited since its last fit: the bic
        # would belong to the previous structure. `make_changes` cannot leave that state — a
        # change that fails to fit is reverted — so the gate had nothing left to catch, and
        # its instruction ("call fit first") named a tool that no longer exists, which would
        # have blocked the finish with no way out.
        # HARD STOP on a gamed noise model: an inflated/collapsed ESTIMATED sigma makes the
        # good-looking bic rest on an implausible noise model (the judge fails those). Refuse the
        # finish with the fix-the-mechanism advice so the agent addresses the FORM, not the noise
        # — but bounded (like the no-fit gate) so a model that genuinely can't reach a plausible
        # sigma still terminates at its honest score instead of spamming finish. A plausible
        # sigma resets the counter, so a real fix immediately clears the block.
        sigma_inflated = bool(fit.get("noise_sigma_inflated"))
        sigma_collapsed = bool(fit.get("noise_sigma_collapsed"))
        if sigma_inflated or sigma_collapsed:
            if state["sigma_refusals"] < _MAX_SIGMA_FINISH_REFUSALS:
                state["armed_token"] = None
                state["sigma_refusals"] += 1
                advice = (fit.get("noise_sigma_advice")
                          or prompts.FINISH_SIGMA_FALLBACK_ADVICE)
                # Name BOTH tails when both are present (one observable inflated, another
                # collapsed) so the agent doesn't fix one and get re-blocked on the other.
                which = " and ".join(w for w, on in (("inflated", sigma_inflated),
                                                     ("collapsed", sigma_collapsed)) if on)
                print(f"[finish BLOCKED {state['sigma_refusals']}/{_MAX_SIGMA_FINISH_REFUSALS}] "
                      f"noise sigma {which} — refusing finish on a gamed noise model")
                return prompts.FINISH_REFUSED_SIGMA_TMPL.format(
                    advice=advice, which=which)
            print(f"[finish RELENTED] noise sigma still implausible after "
                  f"{state['sigma_refusals']} refusals — allowing finish at its honest score")
        else:
            state["sigma_refusals"] = 0
        # HARD STOP on a REPLICATE LACK-OF-FIT: the predictions miss design points by more than
        # those points' OWN replicate scatter allows, which no parameter value can fix — missing
        # STRUCTURE, and the class of run the deleted sigma-ratio arm used to catch by proxy.
        # Keys on the replicate arm alone (never the advisory sign/runs tests) and on its own
        # `available`, so a vacuous / too-thin / untestable verdict never refuses. See
        # _MAX_STRUCTURE_FINISH_REFUSALS.
        lof = ((fit.get("residual_structure") or {}).get("replicate_lack_of_fit") or {})
        if lof.get("available") and lof.get("n_flagged"):
            if state["structure_refusals"] < _MAX_STRUCTURE_FINISH_REFUSALS:
                state["armed_token"] = None
                state["structure_refusals"] += 1
                print(f"[finish BLOCKED {state['structure_refusals']}/"
                      f"{_MAX_STRUCTURE_FINISH_REFUSALS}] replicate lack-of-fit at "
                      f"{lof.get('n_flagged')} design point(s) — refusing finish on a model the "
                      "data's own replicates rule out")
                return prompts.FINISH_REFUSED_REPLICATE_LOF_TMPL.format(
                    n_flagged=lof.get("n_flagged"), n_points=lof.get("n_points"),
                    points=_render_lof_points(lof.get("points")))
            print(f"[finish RELENTED] replicate lack-of-fit still flagged after "
                  f"{state['structure_refusals']} refusals — allowing finish at its honest score")
        else:
            state["structure_refusals"] = 0
        # HARD STOP on an OVER-parameterised model: a finish whose good-looking bic rests on
        # parameters the data cannot constrain — a RANK-DEFICIENT Fisher-information matrix (a
        # non-identifiable parameter COMBINATION the diagnosis surfaces as a near-flat
        # eigen-direction). The sigma gate above is structurally blind to this: extra free parameters
        # keep the per-point fit — and the estimated sigma — tight, so sigma passes while k is
        # padded. Refuse with GENERAL identifiability advice (name the flat-direction params;
        # do NOT hand over the specific fix — the agent must reason the simplification itself),
        # bounded like the sigma gate so a model that genuinely can't be made identifiable still
        # terminates at its honest score. An identifiable fit resets the counter. FAIL OPEN: an
        # unavailable diagnosis (available False, no fim, or the probe returned {}) never
        # refuses — identifiability is best-effort, not a gate a finish can be held hostage to
        # when it can't be computed. Costs a FIM per finish attempt (a few per run) by design.
        #
        # ONE diagnosis call serves this branch and the ARM step below: both halves of it come
        # from the same optimum, and the probe is fetched here (not at the arm) because this
        # branch runs on every finish attempt while arming does not.
        diag = await _current_diagnosis(reader)
        ident = diag.get("identifiability") or {}
        fim_info = ident.get("fim")
        # `stale` means a DIFFERENT STRUCTURE is in the working state than the one this geometry
        # describes (the server labels it rather than re-pricing, which would mean building the
        # current structure). Refusing on it would refuse this finish over another structure's rank,
        # so it fails open. A de-dup no-op is NOT stale — same structure, same optimum, still HEAD's
        # own fit — so the refusal stands there, which matters: that is the one drift a lead can
        # cause inside a run, and it fires precisely BECAUSE the structure did not change.
        if ident.get("available") and not ident.get("stale") and isinstance(fim_info, dict):
            # Gate on RANK DEFICIENCY only (rank < n_estimated), NOT on the mere presence of a
            # flat_direction. Calibration across the reference structures showed the latter
            # over-refuses: a legitimate, identifiable fit routinely carries a flat_direction
            # from ill-conditioning — a parameter pinned at a bound (zero-curvature axis) or a
            # merely correlated group — while its FIM stays full rank (enzyme_kinetics: rank
            # 5/5, two flat directions, correct model). A rank-deficient FIM is the STRONGER
            # signal: a parameter combination the data cannot constrain AT ALL (blasi's 16 free
            # initial conditions on steady-state data, where equilibrium is IC-independent, drop
            # the rank exactly). It is NOT clean either, though — several reference structures
            # are themselves rank-deficient and DO trip this gate (armistead / brannmark / bruno
            # / crauste: parameters pinned at bounds, or genuine collinearity). That is
            # tolerable ONLY because the gate RELENTS after _MAX_PARSIMONY_FINISH_REFUSALS; do
            # NOT drop the relent believing rank deficiency is false-positive-free.
            # flat_directions is still used below — to NAME the offending
            # params — but only rank deficiency triggers the refusal.
            rank_deficient = bool(fim_info.get("rank_deficient"))
            flat_dirs = ident.get("flat_directions") or []
            if rank_deficient:
                if state["parsimony_refusals"] < _MAX_PARSIMONY_FINISH_REFUSALS:
                    state["armed_token"] = None
                    state["parsimony_refusals"] += 1
                    directions = _render_flat_directions(flat_dirs)
                    print(f"[finish BLOCKED {state['parsimony_refusals']}/"
                          f"{_MAX_PARSIMONY_FINISH_REFUSALS}] rank-deficient FIM, flat "
                          f"directions {directions} — refusing finish on an over-parameterised model")
                    return prompts.FINISH_REFUSED_PARSIMONY_TMPL.format(directions=directions)
                print(f"[finish RELENTED] parameters still non-identifiable after "
                      f"{state['parsimony_refusals']} refusals — allowing finish at its "
                      "honest score")
            else:
                state["parsimony_refusals"] = 0
        # Valid, fitted model. First finish call ARMS: show the residual diagnosis and
        # require a confirming second finish. This forces the finish to be a considered call
        # on the fit's residual STRUCTURE (the sign/runs tests plus the replicate lack-of-fit
        # test — the signals that survive
        # estimated sigmas) rather than a reflex. The token identifies THIS fit (bic+nll, plus
        # version as a tiebreak) so an edit + refit between the two calls re-arms rather than
        # confirming a stale diagnosis.
        token = (bic, nll, version)
        if state["armed_token"] != token:
            state["armed_token"] = token
            # The rolled-up verdict rides on the FIT, and is reported nowhere else — the
            # diagnosis carries the per-observable detail behind it, not a second copy.
            rs = fit.get("residual_structure") or {}
            struct = rs.get("summary") or prompts.FINISH_STRUCT_UNAVAILABLE
            per = diag.get("per_observable") or []
            worst = max(per, key=lambda o: abs(o.get("mean_weighted_residual") or 0.0),
                        default=None) if per else None
            worst_line = ""
            if worst is not None:
                st = worst.get("sign_test") or {}
                rtst = worst.get("runs_test") or {}
                worst_line = prompts.FINISH_ARMED_WORST_TMPL.format(
                    observable_id=worst.get("observable_id"),
                    sign_p=st.get("p"), runs_p=rtst.get("p"))
            # The run's OUTSTANDING invalid "does adding structure help?" experiment, if any:
            # a variant that gained parameters yet fit WORSE than its parent, which a nested
            # model cannot do (the server flags it at commit time — see petab_mcp.nesting —
            # and keeps it until a VALID additive experiment supersedes it, so it outlives the
            # step that produced it). Shown here rather than made a refusal: a
            # non-nested variant is legitimate to explore, and the run may have plenty of
            # other grounds to finish. What must not happen is finishing BECAUSE of it.
            invalid = fit.get("invalid_comparison")
            invalid_line = (
                prompts.FINISH_ARMED_INVALID_COMPARISON_TMPL.format(advisory=invalid)
                if invalid else "")
            # The PARAMETER counterpart of the residual verdict: the pairs the data cannot separate
            # whose merges BIC would pay for. Like the residual verdict they ride on the SCORE read
            # (petab_mcp.fitting.fit_summary carries the same `merge_hints` the fit result does), so
            # the gate names the ties the lead already saw instead of projecting new ones.
            merge_line = _render_merge(fit.get("merge_hints"))
            print(f"[finish ARMED v={version}] bic={bic:.6g} — presented diagnosis, "
                  "awaiting confirming finish")
            return prompts.FINISH_ARMED_TMPL.format(
                bic=bic, nll=nll, struct=struct, worst_line=worst_line,
                merge_line=merge_line, invalid_line=invalid_line)
        # Second finish on the same fit -> confirmed. Record it so the driver does not bounce
        # a legitimately-finished run back into the gate.
        state["armed_token"] = None
        finish_state["confirmed"] = True
        chi2_note = ""
        if fit.get("chi2_gof_pvalue") is None and fit.get("reduced_chi2") is None:
            chi2_note = prompts.FINISH_CHI2_UNAVAILABLE_NOTE
        print(f"[finish CONFIRMED v={version}] bic={bic:.6g}")
        return prompts.FINISH_CONFIRMED_TMPL.format(
            bic=bic, nll=nll, chi2_note=chi2_note)

    return StructuredTool.from_function(
        coroutine=finish,
        name="finish",
        description=prompts.FINISH_DESCRIPTION,
    )
