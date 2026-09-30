"""Shared harness for the petab-mcp model-repair examples.

Every example (enzyme kinetics, predator–prey, the repressilator, …) is the same demo
with a different *scenario*: a deliberately mis-specified PEtab problem, an orchestrator
prompt + task, a deterministic canonical repair (the no-LLM ``scripted`` self-test), and
an independent success check. The cleanup → stateful MCP server → deepagents analyst panel
(three analysts + a blinded biocurator) → judge pipeline, plus Phoenix tracing and
the RESULT report, live here and are identical across examples.

An example's ``run.py`` just builds a :class:`Scenario` and calls :func:`run_scenario`.
This is an installed package (``pip install -e .`` from the repo root), so that is a plain
``from gemot import Scenario, run_scenario`` — no ``sys.path`` bootstrap.

Every agent-facing prompt lives in :mod:`gemot.prompts`; the deepagents driver is
:mod:`gemot.agent` and the pipeline + judge + CLI is :mod:`gemot.runner`.

Importing this package is deliberately CHEAP: the names below are resolved lazily (PEP
562), so ``import gemot.agent.dedup`` pulls in no ``mcp``, ``petab_mcp``, phoenix
or langchain. Several submodules here are heavy and the unit tests bind to the light ones;
do NOT add an eager ``from .runner import ...`` back — ``tests/test_import_lightness.py``
guards this.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

from . import paths

# Several submodules here (tracing.py, experiments.py, metrics.py, explorer/bundle.py) import
# from the `phoenix` package. Merely importing it runs `phoenix/config.py`'s module-level
# `ensure_working_dir_if_needed()`, which creates PHOENIX_WORKING_DIR -- defaulting to
# ~/.phoenix -- even when tracing only exports to a remote/self-hosted collector and never
# touches local storage. Default it to a repo-local dir (mirroring slurm/phoenix_data for
# the persistent collector) before any of that can happen, so nothing lands in the home dir.
# Every phoenix-importing module lives inside this package, so this always runs first.
os.environ.setdefault("PHOENIX_WORKING_DIR", str(paths.phoenix_working_dir()))

# Public name -> the submodule that defines it. Resolved on first attribute access.
_LAZY = {
    "run_scenario": ".runner",
    "Scenario": ".scenario",
    "build_true_model": ".scenario",
    "topology_diagnostics": ".scenario",
    "Diagnostics": ".scenario",
    "Baseline": ".scenario",
    "JudgeContext": ".scenario",
    "reaction_rate_law": ".scenario",
    "is_rational": ".scenario",
    "denominator_has_symbol": ".scenario",
    "max_power_in_denominator": ".scenario",
    "max_power_in_numerator": ".scenario",
    "get_fitted_values": ".scenario",
}

if TYPE_CHECKING:  # so IDEs / type checkers still resolve the lazy names
    from .runner import run_scenario
    from .scenario import (
        Baseline,
        build_true_model,
        Diagnostics,
        JudgeContext,
        Scenario,
        denominator_has_symbol,
        get_fitted_values,
        is_rational,
        max_power_in_denominator,
        max_power_in_numerator,
        reaction_rate_law,
        topology_diagnostics,
    )


def __getattr__(name: str):
    """Import the defining submodule on first access (PEP 562)."""
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # cache, so the import runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "run_scenario",
    "Scenario",
    "build_true_model",
    "topology_diagnostics",
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
