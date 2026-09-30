"""Scenario-agnostic runner for the model-repair demos.

Given a :class:`~gemot.scenario.Scenario`, the pipeline is always the same:

  (a) clean the example directory (recreate ``work/`` from the pristine, broken ``problem/``
      copy — this also discards any prior provenance; the SBML-hash-keyed AMICI build cache is
      kept OUTSIDE ``work/`` and reused across runs),
  (b) start the ``petab-mcp`` server on ``work/problem.yaml`` (HTTP transport, one long-lived
      stateful process shared by the whole run), and
  (c) drive the chosen driver — ``deepagents`` (an OpenRouter-hosted model with the analyst
      panel from :mod:`gemot.agent`) or ``scripted`` (the scenario's canonical repair,
      no LLM) — then judge the result INDEPENDENTLY by the run's criterion.

An example's ``run.py`` is a thin wrapper: build the ``Scenario`` and call
:func:`run_scenario`.

  :mod:`.endpoints`   the bio-knowledge MCP URLs the biocurator queries (dependency-free)
  :mod:`.clock`       the MCP call handle + the union-of-busy-intervals tool clock
  :mod:`.provenance`  timestamps, code version, reference score, the live-run marker
  :mod:`.scoring`     residual-structure and noise-sigma projections of the fit reports
  :mod:`.judge`       the BIC verdict, scored on the run's BEST checkpoint
  :mod:`.server`      the petab-mcp server subprocess and its bootstrap
  :mod:`.failure`     degraded results for a run that aborts before a verdict
  :mod:`.drive`       one run: preflight, baseline, driver, judge, archive
  :mod:`.main`        the pipeline around it (cleanup, server, sessions, teardown)
  :mod:`.cli`         the argument parser and :func:`run_scenario`

Most of this package needs ``mcp`` — driving an MCP server is the point of it — but not all of
it: ``endpoints`` imports nothing at all and ``scoring`` needs only ``petab_mcp``. So the names
below resolve LAZILY (PEP 562), exactly as ``gemot.__init__`` does, which keeps
``from gemot.runner.endpoints import BIO_MCP_URLS`` importable in an environment with no
MCP client installed. Do not replace this with eager re-exports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> defining submodule. The private entries are package-internal: the pipeline
# addresses them across module boundaries and the tests have always read them off this package.
_LAZY = {
    "run_scenario": ".cli",
    "build_parser": ".cli",
    "cleanup": ".main",
    "BIO_MCP_URLS": ".endpoints",
    "_main_async": ".main",
    "_drive": ".drive",
    "_judge": ".judge",
    "_restore_best_checkpoint": ".judge",
    "_scoreable_versions": ".judge",
    "_stored_k_and_n": ".judge",
    "_sigma_flagged": ".judge",
    "_SCORING_SCAN_LIMIT": ".judge",
    "_ToolClock": ".clock",
    "_unwrap": ".clock",
    "_make_call": ".clock",
    "_wrap_tool_clock": ".clock",
    "_base_meta": ".provenance",
    "_bic_reference": ".provenance",
    "_git_sha": ".provenance",
    "_now_iso": ".provenance",
    "_write_live_marker": ".provenance",
    "_NOISE_SIGMA_COLLAPSE_FRACTION": ".scoring",
    "_NOISE_SIGMA_BAND_ALPHA": ".scoring",
    "_NOISE_SIGMA_REFERENCE_FACTOR": ".scoring",
    "_NOISE_SIGMA_BAND_MIN_DF": ".scoring",
    "_noise_sigma_flag": ".scoring",
    "_reference_sigma_flag": ".scoring",
    "_residual_structure_fields": ".scoring",
    "_SERVER_BOOTSTRAP": ".server",
    "_free_port": ".server",
    "_wait_until_listening": ".server",
    "_should_reraise": ".failure",
    "_summarize_exc": ".failure",
    "_write_degraded_result": ".failure",
}

if TYPE_CHECKING:  # so IDEs / type checkers still resolve the lazy names
    from .cli import build_parser, run_scenario
    from .endpoints import BIO_MCP_URLS
    from .main import cleanup


def __getattr__(name: str):
    """Import the defining submodule on first access (PEP 562)."""
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # cache, so the import runs once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["run_scenario", "build_parser", "cleanup", "BIO_MCP_URLS"]
