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
      repair, no LLM) — then judge the result INDEPENDENTLY via the criterion gate in ``_judge``
      (``delta_criterion <= scenario.bic_threshold`` against the true model;
      ``scenario.diagnostics`` is report-only).

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
from .. import models


def build_parser(scenario: Scenario) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"{scenario.name} repair demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--driver",
        choices=["deepagents", "scripted", "barebones"],
        default="deepagents",
        help="deepagents: drive an OpenRouter model through the panel + MCP server; "
        "scripted: deterministic repair, no LLM; barebones: the NULL arm — one "
        "stock deepagent with a shell and the PEtab files on disk, no server, no "
        "panel, judged on the problem it delivers (see runner/barebones.py).",
    )
    p.add_argument(
        "--model",
        default=models.DEFAULT,
        help="OpenRouter model id, or a go-to alias (deepagents driver). Aliases: "
        f"{', '.join(models.MODELS)} (see gemot/models.py). Default "
        f"'{models.DEFAULT}' = {models.resolve(models.DEFAULT)}, 1M-token context. "
        "Every aliased model is pinned to a single provider (see "
        "models.provider_routing) so a mirrored copy at another price cannot "
        "silently take the traffic.",
    )
    p.add_argument(
        "--biocurator-model",
        default=None,
        help="Run the BIOCURATOR on its own model (alias or OpenRouter id), leaving "
        "the lead and analysts on --model. It is the panel's heaviest reader — "
        "entity lookups return whole database entries that then ride its context "
        "— and its job is retrieval, not modelling, so a cheaper model here is "
        "worth measuring (e.g. --biocurator-model glm-flash). Default: the "
        "lead's model. Recorded per run as `biocurator_model`.",
    )
    p.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL for the LLM provider (default: OpenRouter).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (default: the OPENROUTER_API_KEY environment variable).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature (deepagents driver); recorded per run for sweeps. A model "
        "that takes none (the Claude 5 family) is sent none.",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="max",
        help="OpenRouter reasoning effort for every panel model (deepagents driver), replacing "
        "the model's own setting (gemot.models); OpenRouter maps it to the nearest level a model "
        "has. `max` (the default) is the largest thinking budget a budget-based model "
        "(Anthropic) is given, ~95%% of its output ceiling. `default` keeps each model's own "
        "setting, and only such a run retries a reasoning runaway at lower effort "
        "(agent.guards). Recorded per run for sweeps.",
    )
    p.add_argument(
        "--recursion-limit",
        type=int,
        default=250,
        help="deepagents/LangGraph super-step budget — a GENEROUS backstop only (above "
        "the longest legitimate ~200-step run). The early-stop chain (non-improving"
        "-streak cap + plateau) does the real stopping, so this should rarely fire.",
    )
    p.add_argument(
        "--n-starts",
        type=int,
        default=50,
        help="pyPESTO multistart count for the `fit` tool (the server's default, so "
        "the agent's own fits use it too unless they pass n_starts).",
    )
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Self-imposed total-runtime budget (seconds): once exceeded, the agent "
        "stops at the next clean step boundary and the judge scores whatever is "
        "currently committed, instead of racing an outer resource limit (e.g. a "
        "Slurm --time allocation) that would hard-kill the process with no "
        "result at all. Default: unlimited (only the stall watchdog + outer "
        "scheduler limit apply). A sweep should set this comfortably below its "
        "Slurm --time (see make_sweep.py). The barebones driver stops its agent "
        "early to compile and score the delivery inside this budget.",
    )
    p.add_argument(
        "--reference-validation",
        choices=["required", "best_effort", "off"],
        default="off",
        help="Server reference-validation mode (off = no network; default for a hermetic demo).",
    )
    # --- blinded `biocurator` analyst: on the deepagents panel every run (endpoints hardcoded in
    #     runner.BIO_MCP_URLS; tools = the curated agent.DEFAULT_BIO_TOOLS allowlist).
    #     Literature access (EuropePMC/bioRxiv) is an ABLATION, not a flag: `PETAB_BIO_LITERATURE=1`,
    #     registered in gemot.ablations alongside every other arm, so it is reachable from
    #     array_task.sh and written into result.json. Adding a flag for it here would put the arm
    #     out of reach of a batch job and give it a second, unrecorded way to be set.
    #     Run `python -m gemot.ablations` to see the arms. ---
    # Phoenix tracing is MANDATORY — every run must be traced, so there is deliberately no flag to
    # disable it (the --phoenix-* options below only configure it, they never turn it off).
    default_project = f"petab-mcp-{scenario.name}"
    p.add_argument(
        "--phoenix-project",
        default=os.environ.get("PHOENIX_PROJECT_NAME", default_project),
        help="Phoenix project name to group this run's traces under.",
    )
    p.add_argument(
        "--phoenix-keep-alive",
        action="store_true",
        help="After the run, keep the process (and any locally launched Phoenix app) "
        "alive until Ctrl-C so the traces stay browsable.",
    )
    p.set_defaults(phoenix=True)
    # --- Weights & Biases metrics logging (off by default; the sweep's analysis UI) ---
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log this run's outcome (pass / ΔBIC / χ² / timings + config) to "
        "Weights & Biases. Orthogonal to Phoenix tracing — use both: W&B for "
        "the sweep metrics UI, Phoenix for the agent trace.",
    )
    p.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "petab-mcp-sweep"),
        help="W&B project to log runs under (default: $WANDB_PROJECT or petab-mcp-sweep).",
    )
    return p


def run_scenario(scenario: Scenario, argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline for ``scenario``, and return a process exit code."""
    args = build_parser(scenario).parse_args(argv)
    args.model = models.resolve(args.model)  # go-to alias (e.g. "luna") -> OpenRouter id
    result: dict = {}
    # The null arm inverts the pipeline (agent first, server after, on what it delivered), so it
    # is a separate main rather than a branch inside the shared one. Resolved lazily: the module
    # imports the agent driver, which a scripted run never needs.
    if args.driver == "barebones":
        from .barebones import _main_async_barebones as _main
    else:
        _main = _main_async
    try:
        exit_code, result = asyncio.run(_main(scenario, args))
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
