"""The deepagents panel: who the lead may delegate to, and what the lead itself may not do.

Why several analysts instead of one:

* **mathematician** — equation structure, dynamical-systems / chemical-reaction-network
  theory, and the numerics of parameter estimation. From the measured dynamics it reasons
  about what reaction-network FORM could generate them; once a model exists it reads the
  equations, the parameter search space, and the pyPESTO/Fides *fitting log* to judge whether
  a misfit is structural (a rate-law FORM that cannot produce the observed dynamics, a broken
  conservation law) or numerical (non-identifiability, a start pinned at a bound).
* **systems_scientist** — mechanistic plausibility. From what is measured and how the data
  behaves it proposes the mechanism a domain expert would write (the species and their
  interactions); once a model exists it judges whether that mechanism is how the system
  actually works rather than a curve-fitting artifact.
* **signal_engineer** — the trajectories as *signals*. Before a model exists it characterizes
  the measured signal itself (shape, periodicity, timescales — the fingerprint the mechanism
  must reproduce); once a model is fitted it reads the measured-vs-simulated residual
  breakdown and reports *where* and *how* the fit fails (bias, drift, shape/timing mismatch)
  without claiming a mechanism.

The personas (in :mod:`petab_harness.prompts`) teach an analyst HOW to reason about building and
refining a dynamic model — its lens, its tools, and the report it must return — but deliberately
give NO hints about any particular problem's answer (no named rate-law forms, no data
fingerprints); the panel has to discover the mechanism from the data. The lead's own prompt and
task are generic for the same reason.

Each analyst gets only the tools its lens needs (:mod:`.tools`), so its context is not bloated
with the full tool surface and its report stays in character. The union of all analyst tools plus
the lead's own editing/recovery tools is what reaches ``create_deep_agent``.

Also here: the lead's CONTAINMENT. deepagents auto-inserts a ``general-purpose`` subagent holding
the full tool pool, which let the lead delegate model authoring into a subagent; and it hands the
lead filesystem builtins for a filesystem that is empty by design. Both are turned off.

The blinded biocurator, whose tools are discovered at runtime, is built in :mod:`.biocurator`.
"""

from __future__ import annotations

from typing import Any

from .. import prompts
from ..prompts import (
    MATHEMATICIAN_PROMPT,
    SIGNAL_ENGINEER_PROMPT,
    SYSTEMS_SCIENTIST_PROMPT,
)
from .guards import _reasoning_runaway_guard, _subagent_turn_limit
from .tools import (
    MATHEMATICIAN_TOOLS,
    SIGNAL_ENGINEER_TOOLS,
    SYSTEMS_SCIENTIST_TOOLS,
)

# Model calls one analyst delegation may make before it must report.
#
# Measured, not guessed: 1,442 delegations across the 239 productive runs of the 260-run main
# sweep (Phoenix spans; slurm/trace_review/subagent_turns.py). Turns per delegation --
#
#     systems_scientist  med 3  p95  7  max 10
#     mathematician      med 3  p95  8  max 20
#     signal_engineer    med 5  p95 12  max 25
#
# signal_engineer sets the bound: it walks per-observable residual SQL, so it is legitimately
# the longest, and a cap near its p95 would truncate ~1 delegation in 20 of exactly the analysis
# worth having. 20 covers its p99 and every other analyst's maximum, trimming only the top ~1%
# of its tail -- which still returns a report, via the wrap-up turn, just earlier than it liked.
#
# Note this is NOT the runaway guard: 4 subagents x a 250-step parent limit is still large in
# principle. This keeps one delegation from wandering; driver._MAX_RUN_TOKENS bounds the run.
ANALYST_MAX_TURNS = 20

# deepagents subagent specs. `subagent_type` values the orchestrator passes to the `task`
# tool are the `name`s here. Each names a curated tool subset (resolved from ALLOWED_TOOLS).
ANALYSTS = [
    {
        "name": "mathematician",
        "description": prompts.MATHEMATICIAN_DESCRIPTION,
        "prompt": MATHEMATICIAN_PROMPT,
        "tools": MATHEMATICIAN_TOOLS,
    },
    {
        "name": "systems_scientist",
        "description": prompts.SYSTEMS_SCIENTIST_DESCRIPTION,
        "prompt": SYSTEMS_SCIENTIST_PROMPT,
        "tools": SYSTEMS_SCIENTIST_TOOLS,
    },
    {
        "name": "signal_engineer",
        "description": prompts.SIGNAL_ENGINEER_DESCRIPTION,
        "prompt": SIGNAL_ENGINEER_PROMPT,
        "tools": SIGNAL_ENGINEER_TOOLS,
    },
]

# ONE source of truth for the active roster — the panel builder and SUBAGENT_NAMES both read
# this, so they cannot disagree about who is on the panel. (The draft-era model_editor rode
# here arm-gated; the diff_edits arm that replaced it changes only the LEAD's own surface, so
# the roster is now arm-independent.)
# Bound to the SAME list object, not a copy: `list(ANALYSTS)` made this a snapshot, so appending
# a persona to ANALYSTS after import reached the panel builder through one name and not the other
# -- the opposite of the single source of truth this exists to be.
ACTIVE_SPECS = ANALYSTS


def _subagent_prompt_key() -> str:
    """The key deepagents' SubAgent spec uses for the persona prompt.

    deepagents renamed the subagent prompt field ``prompt`` -> ``system_prompt`` (mirroring
    the top-level ``create_deep_agent`` rename). Detect it from the ``SubAgent`` type so the
    panel works across releases; default to the current ``system_prompt``.
    """
    try:
        from deepagents.middleware.subagents import SubAgent

        keys = set(getattr(SubAgent, "__annotations__", {}) or {})
        if "system_prompt" not in keys and "prompt" in keys:
            return "prompt"
    except Exception:  # pragma: no cover - defensive
        pass
    return "system_prompt"


def _build_subagents(tools_by_name: dict[str, Any],
                     stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Analyst specs with each persona's tool NAMES resolved to the server's tool OBJECTS.

    deepagents' ``SubAgent`` spec takes ``tools`` as a sequence of ``BaseTool``/callable
    objects (not name strings — older releases resolved names, current ones do not), so we
    map each analyst's curated name list to the actual loaded MCP tool objects, dropping any
    the server does not expose.
    """
    prompt_key = _subagent_prompt_key()
    subagents: list[dict[str, Any]] = []
    for spec in ACTIVE_SPECS:
        tools = [tools_by_name[n] for n in spec["tools"] if n in tools_by_name]
        dropped = [n for n in spec["tools"] if n not in tools_by_name]
        if dropped:
            print(f"[agent] WARNING: analyst {spec['name']!r} missing tools {dropped} "
                  "(dropped from its tool set)")
        subagents.append({
            "name": spec["name"],
            "description": spec["description"],
            prompt_key: spec["prompt"],
            "tools": tools,
            # Bound each delegation. Without this an analyst may explore until the parent's
            # limit or the credit runs out -- see the note on _subagent_turn_limit.
            #
            # The runaway guard rides along because deepagents builds each subagent its OWN
            # middleware stack (graph.py: `spec.get("middleware", [])`) -- the lead's list does
            # not reach here. A reasoning runaway is a property of the MODEL, not of the lead:
            # of 32 length-terminated calls measured on glm-5.3-flash over bachmann/isensee/
            # boehm, 11 were on analysts. Un-guarded, those surface as a blank `task` return
            # that _empty_report_guard papers over with a marker, so the same defect was being
            # diagnosed as an "empty report" instead of a truncation that a cheap reduced-effort
            # retry recovers.
            "middleware": ([_subagent_turn_limit(ANALYST_MAX_TURNS)]
                           + ([_reasoning_runaway_guard(stats)] if stats is not None else [])),
        })
    return subagents


# --- drop deepagents' auto-added `general-purpose` subagent -------------------------------
# On top of the subagents we pass, deepagents AUTO-INSERTS one named `general-purpose` holding
# the FULL tool pool (deepagents 0.6.12 graph.py: it is added unless the active HarnessProfile
# sets `general_purpose_subagent.enabled = False`, or the caller supplies a spec of that name).
# We never curated it, and it broke the panel two ways (issue #112, sweep 50751606):
#   * containment leak — the lead delegated to it and it called `set_model` (9x), `fit` and
#     `change_parameters` INSIDE a subagent, bypassing lead-only authoring;
#   * waste sink — 84 of its 215 tool calls (39%) were filesystem builtins (glob/ls/read_file/
#     grep), which have no role in a PEtab modeling task.
# This is a different mechanism from the tool-pool filter below: the SUBAGENT is removed, not
# its tools. We register a provider-level HarnessProfile because deepagents resolves the
# profile for a PRE-BUILT model (which is what we hand it) via the model's `ls_provider` —
# "openai" for ChatOpenAI, which every model in this harness runs through (OpenRouter is
# reached via ChatOpenAI's base_url). Registration is additive and merges with deepagents' own
# per-model built-in profiles, so it does not clobber them.
_gp_subagent_disabled = False


def _disable_general_purpose_subagent() -> bool:
    """Turn off deepagents' auto-added ``general-purpose`` subagent. Idempotent.

    Returns True when the profile was registered. Warns and returns False if the installed
    deepagents doesn't expose the harness-profile API (a version drift) — the run continues
    with the roster deepagents builds, and ``_general_purpose_enabled`` reports the outcome.
    """
    global _gp_subagent_disabled
    if _gp_subagent_disabled:
        return True
    try:
        from deepagents import (GeneralPurposeSubagentProfile, HarnessProfile,
                                register_harness_profile)
        register_harness_profile(
            "openai",
            HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
        )
    except Exception as exc:  # pragma: no cover - guards against a deepagents API drift
        print(f"[agent] WARNING: could not disable deepagents' `general-purpose` subagent "
              f"({type(exc).__name__}: {exc}); the lead may be able to delegate model authoring "
              "to it (issue #112).")
        return False
    _gp_subagent_disabled = True
    return True


def _general_purpose_enabled(llm) -> "bool | None":
    """Would deepagents still auto-add the ``general-purpose`` subagent for ``llm``?

    Resolved through the SAME lookup ``create_deep_agent`` uses, so this answers the question
    the profile registration was meant to settle rather than approximating it. ``None`` means
    the private resolver is gone (API drift) and the answer is unknown.
    """
    try:
        from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
        profile = _harness_profile_for_model(llm, None).general_purpose_subagent
    except Exception:  # pragma: no cover - guards against a deepagents API drift
        return None
    return not (profile is not None and profile.enabled is False)


# deepagents injects a set of built-in filesystem + todo tools (ls/glob/read_file/write_file/
# edit_file/grep/execute/write_todos) into every agent, on top of the `tools` we pass. This
# task has no filesystem — the real tools are the MCP tools — so those built-ins are pure
# overhead: their JSON schemas are re-sent on every turn (the input-token cost this module
# works hard to keep lean, see _trim_descriptions/TOOL_DESCRIPTIONS), and each call the model
# wastes on them also burns a recursion step. `task` (subagent delegation) is NEVER excluded —
# the panel depends on it. We attach this filter to the LEAD only: neither the per-model
# `HarnessProfile.excluded_tools` route nor a per-subagent `middleware` list reliably reaches
# deepagents 0.6.12's declarative analyst subagents (both were verified to still leak FS calls
# there), so the four curated analysts keep their built-ins for now — the lead is the heavy
# tool-caller and is the one worth trimming. Trimming the analysts cleanly needs an upstream
# deepagents fix. Note this caveat no longer covers the biggest FS-call sink: the uncurated
# `general-purpose` subagent is removed outright above, not filtered.
_BUILTIN_TOOLS_TO_HIDE = frozenset({
    "write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
})


def _builtin_tool_filter():
    """A fresh middleware instance that strips ``_BUILTIN_TOOLS_TO_HIDE`` from the model call.

    Mirrors deepagents' own (private) ``_ToolExclusionMiddleware``: it removes the named tools
    from ``request.tools`` before the model sees them, in BOTH the sync and async paths (the
    deepagents driver runs via ``astream``, so ``awrap_model_call`` is the one that fires).
    Defined lazily so importing this module doesn't require langchain to be installed — only
    the deepagents driver, which pulls langchain in, ever calls this.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    def _name(tool) -> str | None:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    class _DropBuiltinTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            request = request.override(
                tools=[t for t in request.tools if _name(t) not in _BUILTIN_TOOLS_TO_HIDE])
            return handler(request)

        async def awrap_model_call(self, request, handler):
            request = request.override(
                tools=[t for t in request.tools if _name(t) not in _BUILTIN_TOOLS_TO_HIDE])
            return await handler(request)

    return _DropBuiltinTools()
