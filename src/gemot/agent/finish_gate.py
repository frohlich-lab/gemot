"""``finish`` — the lead-only control tool, and the only legitimate way to end a run.

Plain iteration needs no call here; the tool exists to GATE a finish. A finish must rest on a
fitted model with a plausible noise sigma (and, where the run is judged by chi2 against GIVEN
noise, on one that passes it, within bounds), and takes a two-call ARM→CONFIRM handshake so a
diagnosis is always shown before a run is allowed to end. The model it judges is the version the
run is scored on (its best-scoring one with a plausible noise sigma, as the judge picks it), which
the gate first restores as HEAD where the lead explored past it. A silent stop is bounced back into this gate by the driver, so it cannot be sidestepped.

Every string this returns to the model lives in :mod:`gemot.prompts`; the counters and
the relent rules are here.
"""

from __future__ import annotations

import math

from .. import prompts
from ..scenario import ADEQUACY_CHI2, REFERENCE_FREE_GOF_ALPHA, chi2_passes
from .probe import _current_diagnosis, _current_fit, _restore_best, _structured

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
# A deficiency made only of directions in the problem's OWN observation or noise parameters
# (:func:`_fixed_table_parameters`) does not count, and such a direction is never named: no model
# edit reaches one. On schwen (gemot#134) every first refusal led with `[km_nExpID3·1.0]`, a shipped
# detector constant whose readout stays far below its lower bound for any model, and all 4 finished
# runs spent 45-100 min narrowing its bounds to get past the gate. Deliberately NOT "flat in the
# reference too": a refusal the published model would also get is not a defect by itself; one that
# no edit can clear is, because it spends the run's budget for nothing.
_MAX_PARSIMONY_FINISH_REFUSALS = 3

# A finish on a fit that FAILS the run's chi2 bar while its noise is GIVEN is REFUSED: the scenario
# is judged by chi2 (``Scenario.adequacy``) and the fit estimated no noise sigma, so chi2 measures
# the mechanism's misfit and nothing else, and the judge will score the run inadequate there. This is
# NOT the case frohlich-lab/gemot_iclr_2026#54 closed: there the sigmas were ESTIMATED, chi2 pinned
# near n, and a chi2 gate sent the panel after an unreachable target, which is why no other run is
# ever refused on chi2.
# Motivated by gemot_iclr_2026 sweep 63372750 (dessauges, chi2 against given SEMs): 4 of 5 leads
# finished at p 2.5e-26 to 0.005 after 5-9% of their wall budget. A refusal is SPENT only by a
# finish that follows a new version since the previous one, so calling finish again on unchanged
# work does not wear the gate down (the pattern in prompts.py: a lead re-calling finish after every
# refusal until the gate relented). It relents after this many spent refusals, so a model that
# cannot pass still ends at its honest, failing score. Deliberately NOT eased late in the wall
# budget, and the lead is told neither the bound nor that it relents (prompts._CHI2_GATE_RULE): a
# lead that knows when the gate gives in works toward that point instead of the model.
# It is tested BEFORE the parsimony gate (gemot#134), as the lead prompt's ADEQUACY BEFORE PARSIMONY
# rule orders the two: the other way round, an inadequate, over-parameterised model was told only
# that some parameters were unconstrained. dessauges 90 (chi2 p 1e-23) then spent its last 11 fits
# on equivalent rewrites until the streak cap ended it; dessauges 94 (p 1e-66) spent 6 removing flat
# directions, and two of those fits collapsed.
_MAX_ADEQUACY_FINISH_REFUSALS = 5

# Deliberately NO refusal on a residual flag, the replicate lack-of-fit arm's included (gemot#97).
# One refused a finish up to three times on the strength of that arm, and sweep round 3 showed the
# flags do not tell a passing run from a failing one: at the scored version 58 of 146 passing runs
# and 14 of 24 failing runs were flagged; every lucarelli finish was refused, the best run's
# (dBIC -545) at 33 condemned points; sneyd's `open_probability` was flagged at every fitted version
# of every run, down to dBIC -146. The published models fail the same tests at their optimum. blasi
# was the only problem where flags lined up with failure (both of its 2 failures, none of 6 passes;
# its +437 run is why the refusal existed). Do not add a flag-keyed refusal back without showing,
# on runs, that it separates; the flags reach the lead as the ARM text's residual verdict.


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
        terms = [
            f"{p.get('id')}·{p.get('weight')}"
            for p in (d.get("participants") or [])[:max_parts]
            if p.get("id")
        ]
        if terms:
            groups.append("[" + ", ".join(terms) + "]")
    if not groups:
        # rank_deficient can be flagged with no bounded flat-direction participants to name. Names
        # the SECTION, not the bare tool: `sections` defaults to the residual half, which carries no
        # flat directions at all, so a lead following this would spend the turn on the wrong payload.
        return "(see get_fit_diagnosis(sections='parameters') for the flat directions)"
    return "; ".join(groups) + ("; …" if len(flat_directions) > max_dirs else "")


#: The ``role`` ``get_parameters`` gives a parameter of the problem's fixed observation model
#: (``petab_mcp.reconcile.parameter_roles``): a noise sigma, or any parameter the observable
#: formulas and their measurement-level ``observableParameters`` resolve to, a nonlinear detector
#: constant included.
_OBSERVATION_ROLES = frozenset({"noise_sigma", "observable_scaling"})


async def _fixed_table_parameters(reader) -> frozenset[str]:
    """The parameters only the problem's fixed tables use: an observation or noise parameter (by
    ``get_parameters``' ``role``) that the model does not declare (``get_model_structure``). No
    model edit reaches one. A model that declares such a parameter uses it in its own math, which
    the server allows beside a rate constant of the reaction's own, and so owns it.

    Read only when a rank-deficient FIM has directions to name. Empty when either read fails or
    has an odd shape, so nothing is left out and the parsimony gate refuses as it did before.
    """
    try:
        rows = _structured(await reader("get_parameters", {}))
        structure = _structured(await reader("get_model_structure", {}))
    except Exception:  # noqa: BLE001 - a probe never breaks the gate
        return frozenset()
    if not isinstance(rows, list) or not isinstance(structure, dict):
        return frozenset()
    declared = structure.get("parameters")
    if not isinstance(declared, list):
        return frozenset()
    observation = {
        str(row["parameterId"])
        for row in rows
        if isinstance(row, dict)
        and row.get("parameterId")
        and row.get("role") in _OBSERVATION_ROLES
    }
    return frozenset(observation - {str(p) for p in declared})


def _out_of_reach(direction, fixed: frozenset[str]) -> bool:
    """Whether every parameter a flat direction names is in ``fixed``. A direction that names
    none is not shown to be, so it stays."""
    if not isinstance(direction, dict):
        return False
    ids = [p.get("id") for p in (direction.get("participants") or []) if isinstance(p, dict)]
    return bool(ids) and all(i in fixed for i in ids)


def _deficiency_out_of_reach(fim_info: dict, flat_directions, fixed: frozenset[str]) -> bool:
    """Whether the FIM's rank deficiency lies only along directions in ``fixed`` parameters.

    The deficient directions are the ``n_estimated - rank`` flat directions with the smallest
    ``|eigenvalue|``: the diagnosis keeps every direction far flatter than its rank tolerance, the
    deficient ones among them, and an ill-conditioned but full-rank direction beside them does not
    count (see the gate). Every deficient direction has to be listed and out of reach; anything
    the payload does not show (a missing rank, fewer directions listed than the deficiency) keeps
    the refusal."""
    if not fixed:
        return False
    try:
        nullity = int(fim_info["n_estimated"]) - int(fim_info["rank"])
    except (KeyError, TypeError, ValueError):
        nullity = len(flat_directions)
    if nullity < 1 or len(flat_directions) < nullity:
        return False

    def flatness(direction) -> float:
        ev = direction.get("eigenvalue") if isinstance(direction, dict) else None
        finite = isinstance(ev, (int, float)) and not isinstance(ev, bool) and math.isfinite(ev)
        return abs(ev) if finite else 0.0

    deficient = sorted(flat_directions, key=flatness)[:nullity]
    return all(_out_of_reach(d, fixed) for d in deficient)


def _render_merge(hints) -> str:
    """The OUTSTANDING MERGES line for the ARM text, or ``""`` when there is no tie to report.

    Read off ``get_fit_summary``, which carries every tie that still stands, where a fit result
    lists only those new since its parent (frohlich-lab/PEtab-MCP#211), so the gate names the pairs
    the lead was shown on that fit or an earlier one. Fails OPEN like every other diagnosis read
    here: a missing field or an odd shape means "no merge to report", never a raised finish — and one
    unreadable row does not silence the readable ones.

    ``dscore`` is NEGATIVE when the merge wins and the template says "LOWERS the score by", so the slot takes
    the MAGNITUDE — through :func:`_dscore_magnitude`, because ``abs(hint["dscore"] or 0.0)`` rendered
    "LOWERS the score by 0.0" for a tie that beats the price by under 5e-4 nll (it rounds to -0.0,
    which is FALSY) while telling the lead the model was not finished. An edit whose stated payoff is
    zero is not an argument; the same ``or`` swallowed a genuinely absent value into the same 0.0.
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
            ties.append(
                prompts.FINISH_ARMED_MERGE_TIE_TMPL.format(
                    parameters=" == ".join(params),
                    dnll=hint.get("dnll"),
                    dscore=_dscore_magnitude(hint.get("dscore")),
                    move=hint.get("move_scaled"),
                )
            )
        except Exception:  # noqa: BLE001 - an advisory line never fails a finish
            continue
    if not ties:
        return ""
    try:
        return prompts.FINISH_ARMED_MERGE_TMPL.format(ties="".join(ties))
    except Exception:  # noqa: BLE001 - inside the fail-open boundary too: a template slot
        return ""  # mistake must cost the advisory line, never every finish in the run


#: How many of ``residual_structure.sigma_heterogeneity`` the ARM text names, smallest ``p`` first:
#: as many as a fit result names (``petab_mcp.data_access._MAX_NAMED``), where ``get_fit_summary``
#: keeps every one.
_MAX_SIGMAS_NAMED = 3


def _render_sigma_heterogeneity(named) -> str:
    """The SIGMA HETEROGENEITY line for the ARM text, or ``""`` when the rollup names no sigma.

    ``named`` is ``get_fit_summary``'s ``residual_structure.sigma_heterogeneity``, rows of
    ``{parameter_id, p, worst_series, fitted_over_data_noise}`` (frohlich-lab/PEtab-MCP#212). Fails
    OPEN like :func:`_render_merge`: an odd shape costs the line, never the finish, and one
    unreadable row does not silence the readable ones.
    """
    if not isinstance(named, (list, tuple)):
        return ""
    rows: list[str] = []
    for row in named:
        if not isinstance(row, dict) or not row.get("parameter_id"):
            continue
        ratio = row.get("fitted_over_data_noise")
        try:
            rows.append(
                prompts.FINISH_ARMED_SIGMA_TMPL.format(
                    parameter_id=row["parameter_id"],
                    p=row.get("p"),
                    worst_series=row.get("worst_series"),
                    ratio="unknown" if ratio is None else ratio,
                )
            )
        except Exception:  # noqa: BLE001 - an advisory line never fails a finish
            continue
    if not rows:
        return ""
    rest = len(rows) - _MAX_SIGMAS_NAMED
    try:
        return prompts.FINISH_ARMED_SIGMA_HETEROGENEITY_TMPL.format(
            sigmas=",".join(rows[:_MAX_SIGMAS_NAMED]),
            more=prompts.FINISH_ARMED_SIGMA_MORE_TMPL.format(n=rest) if rest > 0 else "",
        )
    except Exception:  # noqa: BLE001 - inside the fail-open boundary, as in _render_merge
        return ""


async def _latest_version(reader) -> int | None:
    """The newest version in the store (``list_versions``): the chi2 gate spends a refusal only on a
    finish that follows a new one. ``None`` when unreadable, which counts as new work, so an
    unreadable store can only bring the relent closer."""
    try:
        rows = _structured(await reader("list_versions", {}))
    except Exception:  # noqa: BLE001 - a probe never breaks the gate
        return None
    versions = [
        row["version"]
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and isinstance(row.get("version"), int)
    ]
    return max(versions) if versions else None


def _dscore_magnitude(dscore) -> str:
    """How much score the merge LOWERS, as displayed: the magnitude, ``unknown`` when there is
    none, and a BOUND rather than ``0.0`` when the win is real but below the resolution shown (it is
    ``2*(dnll - price)`` rounded to 3 dp, and the hint only exists where ``dnll < price``)."""
    if not isinstance(dscore, (int, float)) or isinstance(dscore, bool):
        return "unknown"
    magnitude = abs(float(dscore))
    return f"{magnitude:g}" if round(magnitude, 3) else "less than 0.001"


def _finish_tool(session, finish_state=None, probe=None, *, adequacy=None):
    """The lead-only control tool for the one non-default move: ending the run.

    ``probe`` is the driver's UN-TIMED ``await probe(name, args)`` handle; the gate's two internal
    polls — get_fit_summary for the score, get_fit_diagnosis for the residual detail and the FIM —
    go through it so they don't inflate the run's tool_calls / compute_seconds (falling back to the
    timed ``session`` handle when absent).

    Plain iteration needs no call — to keep going the lead simply makes the next ``make_changes``
    edit — so the tool exists only to gate a finish.

    Every call first makes the version the run is scored on HEAD (:func:`probe._restore_best`): the
    gate judges that model, not wherever the lead's last experiment left HEAD, and the reply says so
    where that moved HEAD. On the progress_signals arm's off side the reply says so on every call,
    without naming a version, and the ARM step quotes no score: either would tell the lead which
    version is its best.

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

    It gates on the χ² goodness-of-fit ONLY where ``adequacy`` (the scenario's declared bar,
    :attr:`gemot.scenario.Scenario.adequacy`) is ``chi2`` AND the fit estimated no noise sigma: the
    judge's own chi2 rule (:func:`gemot.scenario.chi2_passes`) decides, bounded by
    ``_MAX_ADEQUACY_FINISH_REFUSALS``. Everywhere else it deliberately does NOT: whenever the
    observable noise sigmas are estimated (as in the benchmarks of
    frohlich-lab/gemot_iclr_2026#54), χ² pins near the data-point count by construction and its
    p-value is uninformative, so gating ``finish`` on p ≥ 0.05 made the panel abandon its
    best-scoring model to chase an unreachable target (frohlich-lab/gemot_iclr_2026#54). Where χ²
    cannot be computed at all, the tool SAYS so rather than surfacing it as a fit rejection.

    ``finish_state`` also collects how often each refusal fired (``refusals``) and how often a
    finish went through on a relent (``relents``), by gate, for the run's result.
    """
    from langchain_core.tools import StructuredTool

    finish_state = finish_state if finish_state is not None else {}
    finish_state.setdefault("confirmed", False)
    finish_state.setdefault("gave_up", False)
    finish_state.setdefault("refusals", {})
    finish_state.setdefault("relents", {})

    def _count(table: str, gate: str) -> None:
        finish_state[table][gate] = finish_state[table].get(gate, 0) + 1

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
    state = {
        "armed_token": None,
        "no_fit_refusals": 0,
        "sigma_refusals": 0,
        "parsimony_refusals": 0,
        # The chi2 gate's refusals still to spend, and the newest version when it last spent one.
        "adequacy_left": _MAX_ADEQUACY_FINISH_REFUSALS,
        "adequacy_mark": None,
    }

    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    # `rationale` is REQUIRED (no default): ending the run must cost a written justification, the
    # deliberate argument the old `action` string used to be. An argless call is a schema error the
    # model is handed back, not a first half of the arm->confirm handshake.
    async def finish(rationale: str) -> str:
        moved = await _restore_best(reader)
        if moved is not None:
            print(f"[finish] restored the best version v{moved[0]} as HEAD (was v{moved[1]})")
        elif prompts.PROGRESS_SIGNALS_ENABLED:
            return await _decide(rationale)
        # Off the progress_signals arm, the note names no version and goes with every call, moved
        # or not, so that whether HEAD was the best is not told either (gemot#156).
        best, head = moved or (None, None)
        note = prompts.FINISH_RESTORED_BEST_TMPL.format(best=best, head=head)
        return note + await _decide(rationale)

    async def _decide(rationale: str) -> str:
        print(f"[lead finish] {rationale}")
        fit = await _current_fit(reader)
        nll = fit.get("neg_log_likelihood")
        bic = fit.get("bic")
        version = fit.get("version")
        # Require a finite score under the criterion the RUN RANKS BY (`criterion_value`, which
        # equals `bic` under the default BIC), not canonical bic. A finite nll is necessary but not
        # sufficient: get_fit_summary reports no score when n_data_points/k are incomplete, and
        # finishing then would report "n/a" while the refusal text promises there is none to
        # report. It has to be the RANKED score because that is the one `record_fit_progress`
        # checkpoints on: gating on bic instead would let a run finish on a fit the session never
        # scored, wherever the two disagree (aicc is undefined at n <= k+1, hqc below n = 3, and
        # bic is finite at both). Falls back to bic for a server predating the configurable
        # criterion — absent, not None, so an unscoreable fit still reads as unscoreable.
        score = fit.get("criterion_value", bic)
        has_fit = _finite(nll) and _finite(score)
        # Block finish when there is no valid fit to finish on (nothing built, a non-finite
        # objective, or no computable score) — a genuine failure, not a goodness-of-fit
        # judgement. We do NOT gate on chi2_gof_pvalue/reduced_chi2 (see the docstring): they
        # are uninformative when the noise sigmas are estimated, and chasing p ≥ 0.05 costs
        # the best-scoring model.
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
                print(
                    f"[finish GAVE UP] no fittable model after {state['no_fit_refusals']} "
                    "attempts — ending the run as no-model-built"
                )
                return prompts.FINISH_GAVE_UP_NO_FIT_TMPL.format(attempts=state["no_fit_refusals"])
            print(
                f"[finish BLOCKED {state['no_fit_refusals']}/{_MAX_NO_FIT_FINISH_ATTEMPTS}] "
                "no valid fit (no model built, non-finite objective, or no score)"
            )
            _count("refusals", "no_fit")
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
                advice = fit.get("noise_sigma_advice") or prompts.FINISH_SIGMA_FALLBACK_ADVICE
                # Name BOTH tails when both are present (one observable inflated, another
                # collapsed) so the agent doesn't fix one and get re-blocked on the other.
                which = " and ".join(
                    w
                    for w, on in (("inflated", sigma_inflated), ("collapsed", sigma_collapsed))
                    if on
                )
                print(
                    f"[finish BLOCKED {state['sigma_refusals']}/{_MAX_SIGMA_FINISH_REFUSALS}] "
                    f"noise sigma {which} — refusing finish on a gamed noise model"
                )
                _count("refusals", "sigma")
                return prompts.FINISH_REFUSED_SIGMA_TMPL.format(advice=advice, which=which)
            print(
                f"[finish RELENTED] noise sigma still implausible after "
                f"{state['sigma_refusals']} refusals — allowing finish at its honest score"
            )
            _count("relents", "sigma")
        else:
            state["sigma_refusals"] = 0
        # HARD STOP on a fit that fails the run's chi2 bar while its noise is GIVEN (see
        # _MAX_ADEQUACY_FINISH_REFUSALS), by the judge's own chi2 rule (scenario.chi2_passes), so
        # the gate refuses exactly the fits the judge would score inadequate. It FAILS OPEN: the
        # rule says None, never refusing, where a sigma was estimated, the noise report is
        # unreadable, or the p-value is missing or not finite. Ahead of the parsimony gate, so an
        # inadequate model hears that first (gemot#134), and needs no FIM to do so.
        # Not on the progress_signals arm's off side, which withholds the p-value it quotes.
        if adequacy == ADEQUACY_CHI2 and prompts.PROGRESS_SIGNALS_ENABLED:
            p = fit.get("chi2_gof_pvalue")
            if chi2_passes(fit.get("noise_sigma_report"), p) is False:
                reason = f"chi2 GoF p={p:.3g} < {REFERENCE_FREE_GOF_ALPHA:g}"
                if state["adequacy_left"] > 0:
                    latest = await _latest_version(reader)
                    if latest is None or latest != state["adequacy_mark"]:
                        state["adequacy_left"] -= 1
                        state["adequacy_mark"] = latest
                    state["armed_token"] = None
                    spent = _MAX_ADEQUACY_FINISH_REFUSALS - state["adequacy_left"]
                    print(
                        f"[finish BLOCKED {spent}/{_MAX_ADEQUACY_FINISH_REFUSALS}] "
                        f"{reason} against given noise — refusing finish on an inadequate model"
                    )
                    _count("refusals", "adequacy")
                    return prompts.FINISH_REFUSED_ADEQUACY_TMPL.format(
                        p=p, alpha=REFERENCE_FREE_GOF_ALPHA
                    )
                print(
                    f"[finish RELENTED] {reason} still, after {_MAX_ADEQUACY_FINISH_REFUSALS} "
                    "refusals — allowing finish at its honest score"
                )
                _count("relents", "adequacy")
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
        # branch runs on every finish attempt that gets this far while arming does not.
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
            # Leave out what no model edit reaches (see _MAX_PARSIMONY_FINISH_REFUSALS): a
            # deficiency lying only in the problem's own observation or noise parameters is no
            # refusal, and a direction carried only by them is not named.
            reachable = flat_dirs
            if rank_deficient and flat_dirs:
                fixed = await _fixed_table_parameters(reader)
                if _deficiency_out_of_reach(fim_info, flat_dirs, fixed):
                    rank_deficient = False
                    print(
                        f"[finish] rank-deficient FIM only along fixed-table parameters "
                        f"{_render_flat_directions(flat_dirs)} — no edit reaches it, not refusing"
                    )
                reachable = [d for d in flat_dirs if not _out_of_reach(d, fixed)]
            if rank_deficient:
                if state["parsimony_refusals"] < _MAX_PARSIMONY_FINISH_REFUSALS:
                    state["armed_token"] = None
                    state["parsimony_refusals"] += 1
                    directions = _render_flat_directions(reachable)
                    print(
                        f"[finish BLOCKED {state['parsimony_refusals']}/"
                        f"{_MAX_PARSIMONY_FINISH_REFUSALS}] rank-deficient FIM, flat "
                        f"directions {directions} — refusing finish on an over-parameterised model"
                    )
                    _count("refusals", "parsimony")
                    return prompts.FINISH_REFUSED_PARSIMONY_TMPL.format(
                        directions=directions,
                        left_out=(
                            prompts.FINISH_REFUSED_PARSIMONY_LEFT_OUT
                            if len(reachable) < len(flat_dirs)
                            else ""
                        ),
                    )
                print(
                    f"[finish RELENTED] parameters still non-identifiable after "
                    f"{state['parsimony_refusals']} refusals — allowing finish at its "
                    "honest score"
                )
                _count("relents", "parsimony")
            else:
                state["parsimony_refusals"] = 0
        # Valid, fitted model. First finish call ARMS: show the bic and the residual diagnosis
        # and require a confirming second finish, so a finish is a considered call rather than a
        # reflex. The residual verdict is shown as WHERE the fit is off, not as grounds to refuse
        # or to keep going (see the note at the top). The token identifies THIS fit (bic+nll, plus
        # version as a tiebreak) so an edit + refit between the two calls re-arms rather than
        # confirming a stale diagnosis.
        token = (bic, nll, version)
        if state["armed_token"] != token:
            state["armed_token"] = token
            # The rolled-up verdict rides on the FIT, and is reported nowhere else — the
            # diagnosis carries the per-observable detail behind it, not a second copy.
            rs = fit.get("residual_structure") or {}
            struct = rs.get("summary") or prompts.FINISH_STRUCT_UNAVAILABLE
            # Where estimated sigmas or scales could have absorbed what the tests look for: the
            # server's caveat on a clean summary, and the sigmas it names as having done so. Both
            # inform the lead's reading of the verdict; neither refuses (see the note at the top).
            caveat = rs.get("structure_tests_caveat")
            caveat = f" {caveat}" if isinstance(caveat, str) and caveat else ""
            sigma_line = _render_sigma_heterogeneity(rs.get("sigma_heterogeneity"))
            per = diag.get("per_observable") or []
            worst = (
                max(per, key=lambda o: abs(o.get("mean_weighted_residual") or 0.0), default=None)
                if per
                else None
            )
            worst_line = ""
            if worst is not None:
                st = worst.get("sign_test") or {}
                rtst = worst.get("runs_test") or {}
                worst_line = prompts.FINISH_ARMED_WORST_TMPL.format(
                    observable_id=worst.get("observable_id"),
                    sign_p=st.get("p"),
                    runs_p=rtst.get("p"),
                )
            # The run's OUTSTANDING invalid "does adding structure help?" experiment, if any:
            # a variant nested in its parent that gained parameters yet fit WORSE, which only an
            # under-converged fit can do (the server flags it at commit time — see
            # petab_mcp.nesting; since PEtab-MCP#276 only a nested child — and keeps it until a
            # VALID additive experiment supersedes it, so it outlives the step that produced it).
            # Shown here rather than made a refusal: the run may have plenty of other grounds to
            # finish. What must not happen is finishing BECAUSE of it.
            invalid = fit.get("invalid_comparison")
            invalid_line = (
                prompts.FINISH_ARMED_INVALID_COMPARISON_TMPL.format(advisory=invalid)
                if invalid
                else ""
            )
            # The PARAMETER counterpart of the residual verdict: the pairs the data cannot separate
            # whose merges BIC would pay for. Like the residual verdict they ride on the SCORE read
            # (petab_mcp.fitting.fit_summary carries every standing `merge_hints` tie, of which a fit
            # result lists only the new), so the gate names the ties the lead already saw instead of
            # projecting new ones.
            merge_line = _render_merge(fit.get("merge_hints"))
            print(
                f"[finish ARMED v={version}] bic={bic:.6g} — presented diagnosis, "
                "awaiting confirming finish"
            )
            return prompts.FINISH_ARMED_TMPL.format(
                bic=bic,
                nll=nll,
                struct=struct,
                caveat=caveat,
                worst_line=worst_line,
                sigma_line=sigma_line,
                merge_line=merge_line,
                invalid_line=invalid_line,
            )
        # Second finish on the same fit -> confirmed. Record it so the driver does not bounce
        # a legitimately-finished run back into the gate.
        state["armed_token"] = None
        finish_state["confirmed"] = True
        chi2_note = ""
        if (
            prompts.PROGRESS_SIGNALS_ENABLED
            and fit.get("chi2_gof_pvalue") is None
            and fit.get("reduced_chi2") is None
        ):
            chi2_note = prompts.FINISH_CHI2_UNAVAILABLE_NOTE
        print(f"[finish CONFIRMED v={version}] bic={bic:.6g}")
        return prompts.FINISH_CONFIRMED_TMPL.format(bic=bic, nll=nll, chi2_note=chi2_note)

    return StructuredTool.from_function(
        coroutine=finish,
        name="finish",
        description=prompts.FINISH_DESCRIPTION,
    )
