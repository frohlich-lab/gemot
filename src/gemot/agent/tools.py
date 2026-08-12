"""Which petab-mcp tools each agent on the panel can see.

Per-persona curation, not a filter for safety. Every tool's JSON schema is re-sent to the model
on EVERY turn, so the tool set is the biggest single lever on the (input-dominated) token cost —
and a lean set also keeps each analyst in character: the signal engineer reasons about
residuals, not equations; the systems scientist about mechanism, not optimizer internals. That
one rationale governs every list below and is not repeated at each of them.

``ALLOWED_TOOLS`` (the union) is all the panel ever sees; ``_select_tools`` resolves it against
what the server actually offers, and ``_trim_descriptions`` swaps each tool's verbose server
docstring for the panel-facing one-liner in :mod:`petab_harness.prompts` — which is therefore
the text the panel actually reads, and the reason every tool here needs an entry there.
"""

from __future__ import annotations

from ..prompts import DATA_REGIMES_ENABLED, TOOL_DESCRIPTIONS

# ---------------------------------------------------------------- tool exposure
# Per the module docstring: each persona's set is as SMALL as its lens allows, and a tool it
# can do without is dropped rather than kept "just in case". The comments below say only WHY
# each persona gets the tools it does.

# Mathematician: structural capability + estimation numerics. The reaction network (rate
# laws) + the parameter search space + the fit metrics + — uniquely — the pyPESTO/Fides log.
MATHEMATICIAN_TOOLS = [
    "get_model_structure",  # species, reactions, rate laws (empty until a model is built)
    "get_measurements",     # the DATA — dynamics/shape to reason about candidate structure
    # The same shape questions, already decided against the noise floor. A feasibility argument is
    # far stronger stated against a computed verdict ("the data REQUIRES a pulse, and this topology
    # cannot make one") than against an eyeballed trace, and this is the analyst whose job is that
    # argument.
    "get_data_regimes",
    "list_observables",     # what is measured
    "get_parameters",       # scales, bounds — identifiability / conditioning
    "get_fit_summary",
    "get_fitting_log",      # pyPESTO/Fides log: multistart spread, convergence, bounds
    "get_skill",            # PEtab semantics behind the read (steady-state identifiability, etc.)
]

# Systems scientist: is the mechanism real? The mechanism as written, what was measured
# (the readout + data shape), and whether the rate magnitudes are plausible.
SYSTEMS_SCIENTIST_TOOLS = [
    "get_model_structure",  # the mechanism as written
    "list_observables",     # what quantity is measured
    "get_measurements",     # the actual data / dynamic behaviour
    # What any mechanism must reproduce, before proposing one — a requirement is better known up
    # front than discovered after a topology has been built that cannot meet it.
    "get_data_regimes",
    "get_parameters",       # are the rate magnitudes plausible?
    "get_skill",            # PEtab semantics behind the read (steady-state identifiability, etc.)
]

# Signal-processing engineer: residuals as signals, entirely via SQL over the data.
SIGNAL_ENGINEER_TOOLS = [
    "list_data_tables",     # the queryable tables (simulations, residuals, ...) + columns
    "query_data",           # SQL over measured-vs-simulated points + residual breakdowns
    # The standing verdicts this analyst would otherwise re-derive in SQL every run, already
    # measured against the noise floor. Its own reads stay the way to answer anything bespoke; this
    # keeps the routine shape/timescale/dose questions off the lottery of whether it thought to ask.
    "get_data_regimes",
    "get_fit_summary",
    "get_skill",            # PEtab semantics behind the read (e.g. all times inf = steady state)
]

# The orchestrator ("modeling lead") orients, delegates analysis to the three analysts, and
# is the ONLY agent that authors the model. It keeps a MINIMAL surface: read the model + score,
# list/inspect versions, author the whole model.
ORCHESTRATOR_TOOLS = [
    # orient — SMALL reads only. The measured DATA itself is never read into the lead's
    # context; the analysts characterize it and hand back summaries (see run_deepagents:
    # the lead's tool list is restricted to exactly this set).
    "describe_problem",     # model_authored flag, sizes, current version — cheap orientation
    "get_model_structure",  # the current model (empty until built)
    "list_observables",     # what is measured (observable -> species) — needed to name species
    "list_conditions",      # per-condition inputs/doses set in the condition table (match by name)
    # What the DATA requires of any model — excursion vs monotone, how many timescales,
    # dose-response steepness, a conserved total — computed from the measurement table alone. It is
    # a lead tool despite the "no bulky data" rule because it returns VERDICTS, not data: 1-9 kB on
    # the shipped examples, no model, no fit, no compile. That bound is enforced, not hoped for —
    # the report is capped at 12 observables and takes an `observables=` filter, because an
    # unscoped report on a 123-observable problem measured 61 kB (~15k tokens) re-sent every turn.
    # Available before the first model exists, which is the point: a structural requirement is
    # better known up front than discovered after ten fits that cannot meet it.
    "get_data_regimes",
    # score / diagnose (both COMPACT + bounded) / history
    # NOTE: get_fit_summary is deliberately NOT here. `make_changes` already returns
    # the whole fit block — metrics, bic, parameters_at_bound, the noise-sigma verdict + advice,
    # and the residual_structure verdict — so for the LEAD, which is the only agent that fits, a
    # summary call reads back what it was just handed. Trace analysis of sweep 50751606 measured 9
    # such calls against 451 fits: near-zero benefit for a tool schema re-sent every turn. The tool
    # stays on the server: the analysts (who never fit) and the harness's own finish gate and
    # final scoring all read the score through it. `describe_problem` covers has_unfit_edits.
    # per-observable bias/chi2 + worst-N points, AND the FIM/multistart geometry saying WHICH
    # params to cut or merge — two sections of one bounded read (`sections`)
    "get_fit_diagnosis",
    "list_versions",        # scored version history — track which structures were tried + scores
    "get_version",          # read a prior version's Antimony + params (to inspect / re-author from)
    # NOTE: restore_version is deliberately NOT a lead tool. It sets HEAD to a prior version, but
    # make_changes with `antimony=` re-authors the whole model and IGNORES HEAD — so a restore
    # immediately followed by a wholesale re-author (the lead's dominant use: 7 of 9 restores in
    # trace analysis of the current tooling, project petab-mcp-blasi) discarded the restore, a
    # wasted super-step. The lead does not need it: the judge scores the BEST-bic version regardless
    # of HEAD (runner/judge.py restores it before scoring), so a worse change never loses the best
    # model. restore_version REMAINS a server tool — the judge itself calls it — it is just not
    # exposed to the lead. Same reasoning that dropped get_fit_summary from this list.
    "get_errors",
    # THE editor. One call authors the whole model (antimony=), sets estimation bounds/scale
    # (parameters=), or both — and always fits, snapshotting one version. It replaces the
    # set_model -> change_parameters -> fit cycle that was 55% of the lead's calls.
    "make_changes",
    # ...and READ them, so the lead can see what it is changing: id, bounds, scale, nominalValue,
    # plus each row's `role` (kinetic / noise_sigma / observable_scaling / initial_condition /
    # exponent / constant) and `status` (estimated / fixed). The prompt directs the lead here to
    # check whether a sigma_* is ESTIMATED, and it is the read that keeps a widening or a re-author
    # honest: a symbol introduced is added ESTIMATED and counts toward k however tight its bounds,
    # and an observation-model factor must never be wired into a rate law. One row per parameter,
    # so it grows with the model — the only lead tool not strictly size-bounded, kept because
    # a bounds edit without it is made blind.
    "get_parameters",
    # The two SKILLS: short, reusable PEtab/Antimony domain knowledge (what the tables mean,
    # what makes a rate law fittable) — NOT a manual for these tools. How this server behaves
    # is the prompt's and each tool description's job, so the two never overlap by subject
    # matter rather than by discipline.
    # This costs tokens rather than saving them: nothing prunes history, so a fetched skill is
    # re-sent every subsequent turn. Both skills are kept short for exactly that reason.
    "list_skills",
    "get_skill",
    # NOTE: the UNBOUNDED data/diagnostic tools (get_measurements, query_data, get_fitting_log,
    # list_data_tables) are deliberately NOT here — they belong to the analyst subagents, since
    # their large results would pile up in the lead's re-sent history (the input-token cost). The
    # lead instead has get_fit_diagnosis (a COMPACT, size-bounded read of both the residuals and
    # the parameter geometry) for cheap self-diagnosis, and DELEGATES the deep dives — full
    # residual SQL, the optimizer log, raw data characterization — to the analysts via `task`.
]

# The ablation arm (see prompts.DATA_REGIMES_ENABLED). Applied to the lists AFTER they are
# defined, so each persona's rationale above reads as the intended design and this is visibly a
# subtraction from it — and so a future tool added to a list cannot forget to be gated.
if not DATA_REGIMES_ENABLED:
    for _lst in (ORCHESTRATOR_TOOLS, MATHEMATICIAN_TOOLS, SYSTEMS_SCIENTIST_TOOLS,
                 SIGNAL_ENGINEER_TOOLS):
        while "get_data_regimes" in _lst:
            _lst.remove("get_data_regimes")


# The full set handed to create_deep_agent: the union of the lead's tools and every
# analyst's tools (deepagents resolves each subagent's `tools` names from this set).
ALLOWED_TOOLS = set(
    ORCHESTRATOR_TOOLS
    + MATHEMATICIAN_TOOLS
    + SYSTEMS_SCIENTIST_TOOLS
    + SIGNAL_ENGINEER_TOOLS
)


def _trim_descriptions(tools: list) -> list:
    """Shorten each panel tool's description (in place) to cut per-turn input tokens.

    Uses the curated one-liner from TOOL_DESCRIPTIONS; for anything unlisted, truncates the
    server description to its first line. Best-effort — a tool whose description can't be set
    (e.g. a frozen model) is left as-is.
    """
    for t in tools:
        name = getattr(t, "name", None)
        short = TOOL_DESCRIPTIONS.get(name)
        if short is None:
            desc = getattr(t, "description", None)
            if isinstance(desc, str):
                short = desc.strip().split("\n", 1)[0]
        if short:
            try:
                t.description = short
            except Exception:  # pragma: no cover - frozen/rejected attribute
                pass
    return tools


def _select_tools(tools: list) -> list:
    """Curate to ALLOWED_TOOLS. Warn (don't fail open) if expected tools are missing.

    Falling back to the full surface on a rename would silently dump the whole tool set
    into every agent's context — the opposite of the intent — so we keep the intersection
    and surface the drift instead.
    """
    available = {getattr(t, "name", None) for t in tools}
    missing = ALLOWED_TOOLS - available
    if missing:
        print(f"[agent] WARNING: expected tools not found on the server: {sorted(missing)} "
              "(tool names may have drifted)")
    selected = [t for t in tools if getattr(t, "name", None) in ALLOWED_TOOLS]
    if not selected:
        raise RuntimeError(
            "None of the curated ALLOWED_TOOLS matched the server's tools; refusing to fall "
            f"back to the full surface. Server tools: {sorted(n for n in available if n)}"
        )
    return selected
