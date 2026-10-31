"""Which petab-mcp tools each agent on the panel can see.

Per-persona curation, not a filter for safety. Every tool's JSON schema is re-sent to the model
on EVERY turn, so the tool set is the biggest single lever on the (input-dominated) token cost —
and a lean set also keeps each analyst in character: the signal engineer reasons about
residuals, not equations; the systems scientist about mechanism, not optimizer internals. That
one rationale governs every list below and is not repeated at each of them.

``ALLOWED_TOOLS`` (the union) is all the panel ever sees; ``_select_tools`` resolves it against
what the server actually offers. The panel reads each tool's description as the server states it
(frohlich-lab/PEtab-MCP#40), recorded through the server's commit; only the biology-blind arm
replaces them, with its anonymised texts (``_blind_descriptions``).
"""

from __future__ import annotations

from ..prompts import (
    BIOLOGY_BLIND_ENABLED,
    CAUSAL_CONSTRAINTS_ENABLED,
    INDRA_STATEMENTS_ENABLED,
    BIOLOGY_BLIND_TOOL_DESCRIPTIONS,
    LEAD_SELF_READ_TOOLS,
)

# ---------------------------------------------------------------- tool exposure
# Per the module docstring: each persona's set is as SMALL as its lens allows, and a tool it
# can do without is dropped rather than kept "just in case". The comments below say only WHY
# each persona gets the tools it does.

# Mathematician: structural capability + estimation numerics. The reaction network (rate
# laws) + the parameter search space + the fit metrics + — uniquely — the pyPESTO/Fides log.
MATHEMATICIAN_TOOLS = [
    "get_model_structure",  # species; reactions and rate laws when asked (empty until built)
    "get_measurements",  # the DATA — dynamics/shape to reason about candidate structure
    "list_observables",  # what is measured
    "get_parameters",  # scales, bounds — identifiability / conditioning
    "get_fit_summary",
    "get_fitting_log",  # pyPESTO/Fides log: multistart spread, convergence, bounds
    # get_skill is NOT here: every analyst fetched the same petab-model-building document once
    # per spawn (203/203, 217/217, 115/115 across the A/B) — a deterministic call is a constant,
    # so the skill is INLINED into the persona prompts (prompts.py) where it caches as prefix.
]

# Systems scientist: is the mechanism real? The mechanism as written, what was measured
# (the readout + data shape), and whether the rate magnitudes are plausible.
SYSTEMS_SCIENTIST_TOOLS = [
    "get_model_structure",  # the mechanism as written
    "list_observables",  # what quantity is measured
    "get_measurements",  # the actual data / dynamic behaviour
    "get_parameters",  # are the rate magnitudes plausible?
]

# Signal-processing engineer: residuals as signals, entirely via SQL over the data.
SIGNAL_ENGINEER_TOOLS = [
    "list_data_tables",  # the queryable tables (simulations, residuals, ...) + columns
    "query_data",  # SQL over measured-vs-simulated points + residual breakdowns
    "get_fit_summary",
]

# The orchestrator ("modeling lead") orients, delegates analysis to the three analysts, and
# is the ONLY agent that authors the model. It keeps a MINIMAL surface: read the model + score,
# list/inspect versions, author the whole model.
ORCHESTRATOR_TOOLS = [
    # orient / score / diagnose / history — SMALL reads only. The measured DATA itself is never
    # read into the lead's context; the analysts characterize it and hand back summaries (see
    # run_deepagents: the lead's tool list is restricted to exactly this set).
    #
    # These five come from `prompts.LEAD_SELF_READ_TOOLS`, which the lead prompt renders its own
    # "you directly read only …" list from — one list, so the prompt cannot name a read the lead
    # does not hold, and a tool dropped from here disappears from the prompt with it. What each is
    # FOR is documented at that tuple.
    *LEAD_SELF_READ_TOOLS,
    "list_conditions",  # per-condition inputs/doses set in the condition table (match by name)
    # The causal_constraints arm's claims, with what makes each hold on the current model. Only
    # on the arm: the server offers it nowhere else, and `_select_tools` warns of a missing tool.
    *(["list_constraints"] if CAUSAL_CONSTRAINTS_ENABLED else []),
    "list_experiments",  # when each condition applies: the periods of every experiment
    # score / diagnose (both COMPACT + bounded) / history
    # NOTE: get_fit_summary is deliberately NOT here. `make_changes` already returns
    # the whole fit block — metrics, bic, parameters_at_bound, the noise-sigma verdict + advice,
    # and the residual_structure verdict — so for the LEAD, which is the only agent that fits, a
    # summary call reads back what it was just handed. Trace analysis of sweep 50751606 measured 9
    # such calls against 451 fits: near-zero benefit for a tool schema re-sent every turn. The tool
    # stays on the server: the analysts (who never fit) and the harness's own finish gate and
    # final scoring all read the score through it. `describe_problem` covers has_unfit_edits.
    "get_version",  # read a prior version's Antimony + params (to inspect / re-author from)
    # NOTE: restore_version is deliberately NOT a lead tool. It sets HEAD to a prior version, but
    # make_changes with `antimony=` re-authors the whole model and IGNORES HEAD — so a restore
    # immediately followed by a wholesale re-author (the lead's dominant use: 7 of 9 restores in
    # trace analysis of the current tooling, project petab-mcp-blasi) discarded the restore, a
    # wasted super-step. The lead does not need it: the judge scores the BEST-bic version regardless
    # of HEAD (runner/judge.py restores it before scoring), so a worse change never loses the best
    # model. restore_version REMAINS a server tool — the judge itself calls it — it is just not
    # exposed to the lead. Same reasoning that dropped get_fit_summary from this list.
    # get_errors / list_skills are NOT here: zero agent calls across all 37 A/B runs (and the
    # earlier sweeps), while their schemas were re-sent every turn. Both remain server tools.
    # THE editor. One call authors the whole model (antimony=), sets estimation bounds/scale
    # (parameters=), or both — and always fits, snapshotting one version. It replaces the
    # set_model -> change_parameters -> fit cycle that was 55% of the lead's calls.
    "make_changes",
    # The indra_statements arm's mechanism library, each entry as the PySB rules it builds, so a
    # row's policy is chosen knowing what it does and a new one is written from a near one
    # (stripped off the arm below).
    "list_mechanisms",
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
    "get_skill",
    # The raw Antimony of the COMMITTED model, which is the text the lead reads when it wants the
    # model AS WRITTEN. `model_edits` addresses reactions by id, so this is no longer required for
    # editing (get_model_structure is the cheaper read for that) — it is kept for reading a prior
    # version's source before a base_version fork, and is a candidate for removal if the traces
    # show it going unused.
    "get_model",
    # NOTE: the BULK data/diagnostic tools (get_measurements, query_data, get_fitting_log,
    # list_data_tables) are deliberately NOT here — they belong to the analyst subagents, since
    # their results would pile up in the lead's re-sent history (the input-token cost). Note the
    # reason is now VOLUME OVER A RUN, not a single unbounded call: get_measurements and
    # query_data are paged (data_access.MAX_PAGE_ROWS), so no one call can blow the window. Many
    # bounded pages still accumulate in a history re-sent every turn, which is what this
    # restriction is about — the analysts read, the lead gets their summaries. The
    # lead instead has get_fit_diagnosis (a COMPACT, size-bounded read of both the residuals and
    # the parameter geometry) for cheap self-diagnosis, and DELEGATES the deep dives — full
    # residual SQL, the optimizer log, raw data characterization — to the analysts via `task`.
]

#: Every curated list, so the arm strip and ALLOWED_TOOLS below cannot disagree about which
#: lists exist. Adding a persona means adding it HERE, not at three call sites.
_TOOLSETS = (
    ORCHESTRATOR_TOOLS,
    MATHEMATICIAN_TOOLS,
    SYSTEMS_SCIENTIST_TOOLS,
    SIGNAL_ENGINEER_TOOLS,
)

# The ablation arms. Applied to the lists AFTER they are defined, so each persona's rationale
# above reads as the intended design and this is visibly a subtraction from it — and so a future
# tool added to a list cannot forget to be gated. One strip helper, not one loop per arm.
_ARM_TOOLS = (
    (not BIOLOGY_BLIND_ENABLED, ("get_skill",)),
    (INDRA_STATEMENTS_ENABLED, ("list_mechanisms",)),
)
for _on, _names in _ARM_TOOLS:
    if _on:
        continue
    for _lst in _TOOLSETS:
        _lst[:] = [n for n in _lst if n not in _names]


# The full set handed to create_deep_agent: the union of the lead's tools and every
# analyst's tools (deepagents resolves each subagent's `tools` names from this set).
# EVERY curated list. Omitting one silently disarms whatever rides on it: `_select_tools`
# keeps only tools in this union, so an omitted list's tools never reach `tools_by_name` and
# `_build_subagents` drops them from the spec with only a stdout print — a whole arm can be
# unreachable with a green test suite (the roster test asserts NAMES). The draft era hit
# exactly this with the editor's list.
ALLOWED_TOOLS = set().union(*_TOOLSETS)


def _blind_descriptions(tools: list) -> list:
    """On the biology-blind arm, replace each panel tool's description (in place) with its
    anonymised text, refusing a tool that has none; off the arm, leave the server's."""
    if not BIOLOGY_BLIND_ENABLED:
        return tools
    for t in tools:
        name = getattr(t, "name", None)
        if name not in BIOLOGY_BLIND_TOOL_DESCRIPTIONS:
            raise ValueError(f"No anonymous description registered for tool {name!r}")
        t.description = BIOLOGY_BLIND_TOOL_DESCRIPTIONS[name]
    return tools


def _check_statements_editor(tools: list) -> None:
    """On the indra_statements arm, refuse a server whose ``make_changes`` is not the
    statements-only editor. The lead is prompted for statements alone and the statements-only
    guard bounces every other route, so a run against the wrong server would spend its budget
    unscored. The editor must also take each row's binding apart from its statement
    (``bindings``), as the lead is prompted to give it: a server from before the two were split
    reads a row without its `| policy` as malformed, so it would refuse every row the lead sends."""
    if not INDRA_STATEMENTS_ENABLED:
        return
    editor = next((t for t in tools if getattr(t, "name", None) == "make_changes"), None)
    schema = getattr(editor, "args_schema", None)
    args = set(schema.get("properties") or {}) if isinstance(schema, dict) else set()
    if "statements" not in args or args & {"antimony", "model_edits"}:
        raise RuntimeError(
            "indra_statements is on, but the server's make_changes is not the statements-only "
            f"editor (its arguments: {sorted(args)}); was it started without an entity map?"
        )
    if "bindings" not in args:
        raise RuntimeError(
            "indra_statements is on, but the server's make_changes takes no `bindings`: it "
            "predates rows and their bindings given apart, which the lead is prompted for; "
            "move the petab-mcp pin past frohlich-lab/PEtab-MCP#135."
        )


def _select_tools(tools: list) -> list:
    """Curate to ALLOWED_TOOLS. Warn (don't fail open) if expected tools are missing.

    Falling back to the full surface on a rename would silently dump the whole tool set
    into every agent's context — the opposite of the intent — so we keep the intersection
    and surface the drift instead.
    """
    available = {getattr(t, "name", None) for t in tools}
    missing = ALLOWED_TOOLS - available
    if missing:
        print(
            f"[agent] WARNING: expected tools not found on the server: {sorted(missing)} "
            "(tool names may have drifted)"
        )
    selected = [t for t in tools if getattr(t, "name", None) in ALLOWED_TOOLS]
    if not selected:
        raise RuntimeError(
            "None of the curated ALLOWED_TOOLS matched the server's tools; refusing to fall "
            f"back to the full surface. Server tools: {sorted(n for n in available if n)}"
        )
    return selected
