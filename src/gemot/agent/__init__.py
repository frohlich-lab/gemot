"""deepagents driver: a panel of independent analyst subagents + a modeling lead.

The panel is **scenario-agnostic** — the same one drives every example in ``examples/``. The
model-analysis phase is split across analysts that each look at the problem through a different
lens and report to a single **orchestrator** ("modeling lead"). deepagents exposes them to the
lead as a ``task`` tool: the lead calls ``task(description=..., subagent_type=...)``, each
analyst runs its own reasoning loop over *its own curated tool subset*, and returns a written
report. The lead synthesizes the reports and is the only agent that edits the model.

  :mod:`.tools`            what each agent on the panel can see, and the description trim
  :mod:`.subagents`        the analyst specs + the lead's containment (no general-purpose, no builtins)
  :mod:`.biocurator`       the blinded bio-knowledge analyst, assembled from runtime-discovered tools
  :mod:`.literature`       provenance for what the biocurator read, and the EuropePMC trim
  :mod:`.probe`            reading the server's CURRENT state through a bare call handle
  :mod:`.finish_gate`      ``make_decision`` — the only legitimate way to end a run
  :mod:`.token_usage`      per-run token accounting across the lead and its subagents
  :mod:`.circuit_breaker`  cut a tool that keeps failing, or whose server has gone away
  :mod:`.dedup`            pure identity/de-dup for proposed tool calls
  :mod:`.guards`           middleware: make proposed tool calls actually FIRE, and arrive intact
  :mod:`.driver`           :func:`run_deepagents` — builds the graph and runs the loop

Every agent-facing string is in :mod:`petab_harness.prompts`, not here; observability is Arize
Phoenix tracing (:mod:`petab_harness.tracing`), which the runner turns on before the agent starts.

Importing this package is CHEAP by design: every module above touches only stdlib at import
time, and each langchain / deepagents / openai import is lazy, inside the function that needs it.
The unit tests bind to these helpers with none of those installed
(``tests/test_import_lightness.py`` guards it), so keep new module-scope imports stdlib-only.

The names re-exported below are package-internal — reachable as ``agent._foo`` because tests and
the driver have always addressed them that way. ``run_deepagents`` is the only public one.
"""

from __future__ import annotations

from . import (
    biocurator,
    circuit_breaker,
    dedup,
    driver,
    finish_gate,
    guards,
    literature,
    probe,
    subagents,
    token_usage,
    tools,
)
from .biocurator import (
    BIOCURATOR_NAME,
    DEFAULT_BIO_TOOLS,
    LITERATURE_BIO_TOOLS,
    SUBAGENT_NAMES,
    _build_biocurator_subagent,
)
from .circuit_breaker import (
    _breaker_message,
    _disabled_tool_filter,
    _is_transport_error,
    _new_breaker_state,
    _record_error,
    _result_error,
    _transport_breaker_message,
    _wrap_tools_with_circuit_breaker,
)
from .dedup import _dedupe_raw_tool_calls, _dedupe_tool_calls, _tool_call_key
from .driver import (
    _MAX_SILENT_BOUNCES,
    run_deepagents,
)
from .finish_gate import (
    _MAX_NO_FIT_FINISH_ATTEMPTS,
    _MAX_SIGMA_FINISH_REFUSALS,
    _decision_tool,
)
from .guards import (
    _MAX_IDLE_MODEL_STEPS,
    _MAX_TASK_ARG_REJECTIONS,
    _task_arg_guard,
    _task_arg_reject_message,
    _tool_dispatch_guard,
    _unexpected_task_kwargs,
)
from .literature import _content_to_text, _trim_europepmc_search, _wrap_literature_recorder
from .probe import _current_diagnosis, _current_fit, _model_committed, _structured
from .subagents import (
    ANALYSTS,
    _build_subagents,
    _builtin_tool_filter,
    _disable_general_purpose_subagent,
    _general_purpose_enabled,
    _subagent_prompt_key,
)
from .token_usage import _token_accountant
from .tools import (
    ALLOWED_TOOLS,
    MATHEMATICIAN_TOOLS,
    ORCHESTRATOR_TOOLS,
    SIGNAL_ENGINEER_TOOLS,
    SYSTEMS_SCIENTIST_TOOLS,
    _select_tools,
    _trim_descriptions,
)

# The prompt constants tests and callers have always read off this module. They live in
# petab_harness.prompts now; re-exported so `agent.BIOCURATOR_PROMPT` still resolves.
from ..prompts import (
    BIOCURATOR_DESCRIPTION,
    BIOCURATOR_PROMPT,
    MATHEMATICIAN_PROMPT,
    SIGNAL_ENGINEER_PROMPT,
    SYSTEMS_SCIENTIST_PROMPT,
    TOOL_DESCRIPTIONS,
    _biocurator_entity_block,
    _IDLE_NUDGE,
    _NO_MODEL_NUDGE,
    _SILENT_FINISH_NUDGE,
)

__all__ = ["run_deepagents"]
