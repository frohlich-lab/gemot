"""``make_decision`` — the lead-only control tool, and the only legitimate way to end a run.

Plain iteration needs no call here; the tool exists to record a rollback and to GATE a finish. A
finish must rest on a current, fitted model with a plausible noise sigma, and takes a two-call
ARM→CONFIRM handshake so a diagnosis is always shown before a run is allowed to end. A silent
stop is bounced back into this gate by the driver, so it cannot be sidestepped.

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


def _decision_tool(session, finish_state=None, probe=None):
    """A lead-only control tool for the two non-default decisions: ``rollback`` and ``finish``.

    ``probe`` is the driver's UN-TIMED ``await probe(name, args)`` handle; the gate's internal
    get_fit_summary / get_fit_diagnosis polls go through it so they don't inflate the run's
    tool_calls / compute_seconds (falling back to the timed ``session`` handle when absent).

    Plain iteration needs no call — to keep going the lead simply makes the next ``make_changes``
    edit — so the tool exists only to record a rollback or to gate a finish.

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

    # `finish` is handled entirely in the branch below (every path there returns), so it needs
    # no entry here — only `rollback` and the unknown-action default ever reach _guidance.
    _guidance = prompts.DECISION_GUIDANCE
    # `armed_token` identifies the fit a finish was ARMED on (the diagnosis has been shown and a
    # confirming second finish is awaited), or None when not armed. The token is the fit's
    # identity, so an edit + refit between the two calls re-arms (the shown diagnosis would
    # otherwise be stale). It is NOT the get_fit_summary "version": that is None after a
    # rollback/restore_version (the persisted per-version sim blob carries no version), and
    # `None != None` would then skip arming and confirm on the first call — bypassing the gate on
    # the common rollback→finish path. bic+nll identify the scored fit whether or not a version
    # is attached.
    state = {"armed_token": None, "no_fit_refusals": 0, "sigma_refusals": 0}

    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    async def make_decision(action: str, rationale: str = "") -> str:
        act = (action or "").strip().lower()
        print(f"[lead decision] {act or '?'} — {rationale}")
        if act == "finish":
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
            # Valid, fitted model. First finish call ARMS: show the residual diagnosis and
            # require a confirming second finish. This forces the finish to be a considered call
            # on the fit's residual STRUCTURE (sign/runs tests — the signal that survives
            # estimated sigmas) rather than a reflex. The token identifies THIS fit (bic+nll,
            # plus version as a tiebreak) so an edit + refit — or a rollback, which leaves
            # version None but changes bic/nll — re-arms rather than confirming a stale diagnosis.
            token = (bic, nll, version)
            if state["armed_token"] != token:
                state["armed_token"] = token
                diag = await _current_diagnosis(reader)
                rs = diag.get("residual_structure") or {}
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
                print(f"[finish ARMED v={version}] bic={bic:.6g} — presented diagnosis, "
                      "awaiting confirming finish")
                return prompts.FINISH_ARMED_TMPL.format(
                    bic=bic, nll=nll, struct=struct, worst_line=worst_line)
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
        # Any non-finish decision (rollback) disarms a pending finish: the state has changed, so
        # a later finish must re-present the diagnosis rather than confirm against a stale one.
        state["armed_token"] = None
        return _guidance.get(act, prompts.DECISION_UNKNOWN_ACTION)

    return StructuredTool.from_function(
        coroutine=make_decision,
        name="make_decision",
        description=prompts.MAKE_DECISION_DESCRIPTION,
    )
