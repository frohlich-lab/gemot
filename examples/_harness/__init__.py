"""Shared harness for the petab-mcp model-repair examples.

Every example (enzyme kinetics, predator–prey, the repressilator, …) is the same demo
with a different *scenario*: a deliberately mis-specified PEtab problem, an orchestrator
prompt + task, a deterministic canonical repair (the no-LLM ``scripted`` self-test), and
an independent success check. The cleanup → stateful MCP server → deepagents analyst panel
(three analysts + a blinded biocurator) → judge pipeline, plus Phoenix tracing and
the RESULT report, live here and are identical across examples.

An example's ``run.py`` just builds a :class:`Scenario` and calls :func:`run_scenario`.
Because ``examples/`` is not an installed package, each ``run.py`` puts this directory on
``sys.path`` (``sys.path.insert(0, str(Path(__file__).resolve().parent.parent))``) before
``from _harness import ...``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Several submodules here (tracing.py, experiments.py, metrics.py, explorer/bundle.py) import
# from the `phoenix` package. Merely importing it runs `phoenix/config.py`'s module-level
# `ensure_working_dir_if_needed()`, which creates PHOENIX_WORKING_DIR -- defaulting to
# ~/.phoenix -- even when tracing only exports to a remote/self-hosted collector and never
# touches local storage. Default it to a repo-local dir (mirroring slurm/phoenix_data for
# the persistent collector) before any of that can happen, so nothing lands in the home dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PHOENIX_WORKING_DIR", str(_REPO_ROOT / ".phoenix"))

from .runner import run_scenario
from .scenario import (
    Baseline,
    Diagnostics,
    JudgeContext,
    Scenario,
    denominator_has_symbol,
    get_fitted_values,
    is_rational,
    max_power_in_denominator,
    max_power_in_numerator,
    reaction_rate_law,
)

__all__ = [
    "run_scenario",
    "Scenario",
    "Diagnostics",
    "Baseline",
    "JudgeContext",
    "reaction_rate_law",
    "is_rational",
    "denominator_has_symbol",
    "max_power_in_denominator",
    "max_power_in_numerator",
    "get_fitted_values",
]
