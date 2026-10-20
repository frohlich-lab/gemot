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
  :mod:`.finish_gate`      ``finish`` — the only legitimate way to end a run
  :mod:`.token_usage`      per-run token accounting across the lead and its subagents
  :mod:`.circuit_breaker`  cut a tool that keeps failing, or whose server has gone away
  :mod:`.dedup`            pure identity/de-dup for proposed tool calls
  :mod:`.guards`           middleware: make proposed tool calls actually FIRE, and arrive intact
  :mod:`.driver`           :func:`run_deepagents` — builds the graph and runs the loop

Every agent-facing string is in :mod:`gemot.prompts`, not here; observability is Arize
Phoenix tracing (:mod:`gemot.tracing`), which the runner turns on before the agent starts.

Importing this package is CHEAP by design: every module above touches only stdlib at import
time, and each langchain / deepagents / openai import is lazy, inside the function that needs it.
The unit tests bind to these helpers with none of those installed
(``tests/test_import_lightness.py`` guards it), so keep new module-scope imports stdlib-only.

The names re-exported below are package-internal — reachable as ``agent._foo`` because tests and
the driver have always addressed them that way. ``run_deepagents`` is the only public one.

Re-export ONLY what is actually read through this facade. The block used to list 58 names, 22 of
which nothing ever reached this way — each had an in-package importer or a test importing its
defining submodule directly. A name here that no one resolves off the package is not an
interface, just a line to keep in sync.

Careful when trimming further: several tests bind the package to a local (``mod = _agent``,
``tests/test_finish_gate.py``) before reaching names off it, so grepping for ``agent.NAME``
UNDER-counts what is live. The test suite is the only reliable check.
"""

from __future__ import annotations

# Bind the submodules so `agent.dedup` / `agent.literature` resolve as attributes (tests reach
# them that way). Each touches only stdlib at import time, so this stays import-light.
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
    DEFAULT_BIO_TOOLS,
    LITERATURE_BIO_TOOLS,
    SUBAGENT_NAMES,
    _build_biocurator_subagent,
)
from .circuit_breaker import (
    _is_transport_error,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
)
from .dedup import (
    _dedupe_raw_tool_calls,
    _dedupe_tool_calls,
    _tool_call_key,
)
from .driver import _MAX_SILENT_BOUNCES, run_deepagents
from .finish_gate import (
    _MAX_NO_FIT_FINISH_ATTEMPTS,
    _MAX_PARSIMONY_FINISH_REFUSALS,
    _MAX_SIGMA_FINISH_REFUSALS,
    _MAX_STRUCTURE_FINISH_REFUSALS,
    _finish_tool,
)
from .guards import (
    _empty_report_guard,
    _is_blank_report,
    _report_text,
    _task_arg_guard,
    _task_arg_reject_message,
    _tool_dispatch_guard,
    _unexpected_task_kwargs,
)
from .literature import _trim_europepmc_search, _wrap_literature_recorder
from .probe import _model_committed
from .subagents import (
    ACTIVE_SPECS,
    ANALYSTS,
    _build_subagents,
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
)

# The prompt constants tests and callers have always read off this module. They live in
# gemot.prompts now; re-exported so `agent.BIOCURATOR_PROMPT` still resolves.
from ..prompts import (
    BIOCURATOR_PROMPT,
    SUBAGENT_EMPTY_REPORT_MARKER,
    SUBAGENT_EMPTY_REPORT_NUDGE,
    _biocurator_entity_block,
    _NO_MODEL_NUDGE,
    _SILENT_FINISH_NUDGE,
)

__all__ = ["run_deepagents"]
