#!/usr/bin/env python3
"""Scenario-agnostic runner for the model-repair demos.

Given a :class:`~scenario.Scenario`, the pipeline is always the same:

  (a) clean the example directory (recreate ``work/`` from the pristine, broken
      ``problem/`` copy — this also discards any prior provenance; the SBML-hash-keyed
      AMICI build cache is kept OUTSIDE ``work/`` and reused across runs),
  (b) start the ``petab-mcp`` server on ``work/problem.yaml`` (HTTP transport, one
      long-lived stateful process shared by the whole run), and
  (c) drive the chosen driver — ``deepagents`` (an OpenRouter-hosted model with the
      analyst panel from :mod:`agent` — three analysts + the biocurator) or
      ``scripted`` (the scenario's canonical
      repair, no LLM) — then judge the result INDEPENDENTLY via the BIC gate in ``_judge``
      (``delta_bic <= scenario.bic_threshold`` against the true model; ``scenario.diagnostics``
      is report-only).

HTTP (not stdio) is used because AMICI's C++ build toolchain writes to the server's
stdout while compiling, which would corrupt a stdio JSON-RPC channel; that stdout is
captured to ``work/amici_build.log`` while uvicorn/Python logging goes to
``work/server.log``.

An example's ``run.py`` is a thin wrapper: build the ``Scenario`` and call
:func:`run_scenario`.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from .failure import _should_reraise, _summarize_exc
from .main import _main_async
from ..scenario import Scenario

def build_parser(scenario: Scenario) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{scenario.name} repair demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--driver", choices=["deepagents", "scripted"], default="deepagents",
                   help="deepagents: drive an OpenRouter model; scripted: deterministic repair, no LLM.")
    p.add_argument("--model", default="deepseek/deepseek-v4-pro",
                   help="OpenRouter model id (deepagents driver). Default v4-pro: on the "
                        "repressilator it passes 2/3 near-parsimoniously vs v4-flash's lucky "
                        "1/5, at ~$0.58/run (it thrashes less).")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible base URL for the LLM provider (default: OpenRouter).")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter API key (default: the OPENROUTER_API_KEY environment variable).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature (deepagents driver); recorded per run for sweeps.")
    p.add_argument("--recursion-limit", type=int, default=250,
                   help="deepagents/LangGraph super-step budget — a GENEROUS backstop only (above "
                        "the longest legitimate ~200-step run). The early-stop chain (non-improving"
                        "-streak cap + plateau) does the real stopping, so this should rarely fire.")
    p.add_argument("--n-starts", type=int, default=50,
                   help="pyPESTO multistart count for the `fit` tool (the server's default, so "
                        "the agent's own fits use it too unless they pass n_starts).")
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="Self-imposed total-runtime budget (seconds): once exceeded, the agent "
                        "stops at the next clean step boundary and the judge scores whatever is "
                        "currently committed, instead of racing an outer resource limit (e.g. a "
                        "Slurm --time allocation) that would hard-kill the process with no "
                        "result at all. Default: unlimited (only the stall watchdog + outer "
                        "scheduler limit apply). A sweep should set this comfortably below its "
                        "Slurm --time (see make_sweep.py).")
    p.add_argument("--reference-validation", choices=["required", "best_effort", "off"], default="off",
                   help="Server reference-validation mode (off = no network; default for a hermetic demo).")
    # --- blinded `biocurator` analyst: on the deepagents panel every run (endpoints hardcoded in
    #     runner.BIO_MCP_URLS). --bio-literature adds EuropePMC/bioRxiv; --bio-tools curates. ---
    p.add_argument("--bio-literature", action="store_true",
                   help="Also give the biocurator EuropePMC + bioRxiv LITERATURE search (OFF by "
                        "default). Its structured knowledge (interactions/pathways/ontologies/"
                        "OmniPath) is included by default and is general biology; literature search over a "
                        "benchmark problem's distinctive entity panel could surface its withheld "
                        "source paper, so it is opt-in. Unambiguously useful on NOVEL problems.")
    p.add_argument("--bio-tools", default=os.environ.get("BIO_MCP_TOOLS", ""),
                   help="Which bio-server tools the biocurator gets. Default (empty) = the CURATED "
                        "set (agent.DEFAULT_BIO_TOOLS: identity + domains + interactions + pathways "
                        "+ GO + ontology + OmniPath), plus EuropePMC/bioRxiv when --bio-literature "
                        "is set — a deliberate lean set, since the server exposes ~50 tools and "
                        "every schema is re-sent each turn. Pass a comma-separated list of tool "
                        "NAMES to choose your own set (overrides --bio-literature), or 'all' to "
                        "expose every discovered tool. If the curated set matches no tool on the "
                        "connected server(s), all discovered tools are exposed (warned).")
    # --- Arize Phoenix agent tracing (on by default; deepagents driver only) ---
    p.add_argument("--no-phoenix", dest="phoenix", action="store_false",
                   help="Disable Arize Phoenix tracing (it is ON by default).")
    default_project = scenario.phoenix_project or f"petab-mcp-{scenario.name}"
    p.add_argument("--phoenix-project", default=os.environ.get("PHOENIX_PROJECT_NAME", default_project),
                   help="Phoenix project name to group this run's traces under.")
    p.add_argument("--phoenix-keep-alive", action="store_true",
                   help="After the run, keep the process (and any locally launched Phoenix app) "
                        "alive until Ctrl-C so the traces stay browsable.")
    p.set_defaults(phoenix=True)
    # --- Weights & Biases metrics logging (off by default; the sweep's analysis UI) ---
    p.add_argument("--wandb", action="store_true",
                   help="Log this run's outcome (pass / ΔBIC / χ² / timings + config) to "
                        "Weights & Biases. Orthogonal to Phoenix tracing — use both: W&B for "
                        "the sweep metrics UI, Phoenix for the agent trace.")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "petab-mcp-sweep"),
                   help="W&B project to log runs under (default: $WANDB_PROJECT or petab-mcp-sweep).")
    return p


def run_scenario(scenario: Scenario, argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline for ``scenario``, and return a process exit code."""
    args = build_parser(scenario).parse_args(argv)
    result: dict = {}
    try:
        exit_code, result = asyncio.run(_main_async(scenario, args))
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        # A teardown-time failure (e.g. an anyio TaskGroup BaseExceptionGroup from the MCP
        # client after a provider stall — a BaseException, not an Exception, so it slipped the
        # old `except Exception`) can escape _main_async AFTER _drive already wrote
        # result.json. Exit cleanly (nonzero) instead of crash-tracebacking — the on-disk
        # result.json, if written, is the source of truth the batch/aggregator reads.
        # _should_reraise still lets a genuine interrupt / SystemExit / top-level cancellation
        # propagate untouched.
        if _should_reraise(exc):
            raise
        print(f"[run] exited with an unrecovered teardown error ({_summarize_exc(exc)}).")
        return 1
    # W&B logging is a post-run, best-effort step (covers run.py direct AND the sweep's
    # run.py subprocesses) — never let it change the exit code.
    if getattr(args, "wandb", False):
        try:
            from ..wandb_logging import log_result
            log_result(result, project=args.wandb_project)
        except Exception as exc:  # pragma: no cover - telemetry must not break the run
            print(f"[wandb] skipped ({type(exc).__name__}: {exc})")
    return exit_code
