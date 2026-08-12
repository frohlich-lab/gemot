"""One run, from a connected session to a written verdict.

Preflight the server, score the baseline, dispatch to the chosen driver (the deepagents panel or
the scenario's scripted canonical repair), then judge and archive. Wrapped so that anything
escaping the driver still produces a degraded result rather than a traceback and no record.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import os
import subprocess
import time
from pathlib import Path

from mcp import ClientSession

from .. import ablations
from .failure import _should_reraise, _summarize_exc, _write_degraded_result
from .judge import _judge
from .clock import _make_call, _wrap_tool_clock
from .provenance import _base_meta, _write_live_marker
from ..scenario import Baseline, Scenario

async def _drive(scenario: Scenario, session: ClientSession, work: Path,
                 args: argparse.Namespace, run_started_at: str | None = None,
                 server_proc: "subprocess.Popen | None" = None,
                 bio_sessions: "list | None" = None) -> "tuple[int, dict]":
    """Preflight, run the chosen driver, and judge — all on one live session.

    ``run_started_at`` (the real start captured by ``_main_async`` before the server launched)
    keeps a degraded result's start time truthful; ``server_proc`` lets an abort record whether
    the MCP server had crashed (see :func:`_write_degraded_result`).

    Returns ``(exit_code, result)`` (see :func:`_judge`).
    """
    t0 = time.monotonic()
    # Capture the un-timed call handle BEFORE wrapping, so the agent's internal early-stop polls
    # (get_progress, once per super-step) don't inflate the tool clock / tool_calls metric.
    raw_call_tool = session.call_tool
    clock = _wrap_tool_clock(session)  # split server/compute time from LLM time
    call = _make_call(session)

    # Stamp the run's start + provenance up front (before the slow baseline compile) so a
    # degraded result.json is still well-formed if preflight/baseline itself blows up.
    scripted = args.driver == "scripted"
    base_meta = _base_meta(scenario, args, run_started_at)
    run_started_at = base_meta["run_started_at"]
    git_sha = base_meta["git_sha"]

    tracing = None
    try:
        desc = await call("describe_problem")
        print(f"[preflight] model '{desc.get('model_id')}' | "
              f"reactions={desc.get('n_reactions_detail')} | "
              f"observables={desc.get('observables')} | "
              f"free params={desc.get('free_parameters')}")

        # Record the initial species set so the judge can reject "repairs" that invent a new,
        # unobserved species to bend the curve rather than fixing the mechanism.
        baseline_structure = await call("get_model_structure")
        baseline_species = set(baseline_structure.get("species", []) or [])

        if desc.get("model_authored", True):
            print("[preflight] scoring baseline (first AMICI compile is slow)...")
            baseline_fit = await call("get_fit_summary")
            # Guard the baseline scoring float(): a shipped model whose FIRST fit fails returns
            # neg_log_likelihood=None (or a non-finite value). Coerce None -> NaN rather than
            # crashing the whole run on float(None), then treat any non-finite baseline as "no
            # usable baseline" (scored like a zero-shot build). A single failed first fit is not
            # an infrastructure failure — the run should still reach the agent + judge, which
            # score against the true-model reference and don't need a baseline nll.
            raw_bl = baseline_fit.get("neg_log_likelihood") if isinstance(baseline_fit, dict) else None
            try:
                bl_nll = float(raw_bl)
            except (TypeError, ValueError):
                bl_nll = float("nan")
            if not math.isfinite(bl_nll):
                bl_nll = None
            baseline = Baseline(nll=bl_nll, species=baseline_species, structure=baseline_structure)
            _bl_disp = f"{baseline.nll:.6g}" if baseline.nll is not None else "n/a (first fit failed)"
            print(f"[preflight] baseline neg_log_likelihood = {_bl_disp}")
        else:
            # Zero-shot: the problem ships without a model — nothing to score until the agent
            # builds one via make_changes. The judge compares to the true-model reference, not a
            # baseline, so a None baseline is fine.
            print("[preflight] no model shipped — zero-shot build (no baseline to score)")
            baseline = Baseline(nll=None, species=baseline_species, structure=baseline_structure)

        # --- scripted driver: deterministic canonical repair, no LLM, no tracing ----------
        if scripted:
            print(f"[scripted] applying canonical fix: {scenario.fix_summary or '(see scenario)'}")
            await scenario.scripted_fix(call, args.n_starts)
            meta = {**base_meta, "stop_reason": "completed"}
            return await _judge(scenario, call, baseline, work, meta, clock, t0)

        # --- deepagents driver: the LLM panel diagnoses + repairs --------------------------
        # Arize Phoenix tracing (mandatory — always on, no off switch): instrument the
        # LangChain stack BEFORE the agent runs so every LLM + tool-call span is captured.
        if args.phoenix:
            from ..tracing import setup_phoenix_tracing

            tracing = setup_phoenix_tracing(project_name=args.phoenix_project)

        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "No OpenRouter API key: set OPENROUTER_API_KEY (or pass --api-key). "
                "Get one at https://openrouter.ai/keys, or use --driver scripted for a "
                "no-LLM self-test."
            )
        print(f"[agent] driving {args.model} via OpenRouter + deepagents "
              f"(recursion_limit={args.recursion_limit})")
        from ..agent import run_deepagents
        from ..tracing import record_run_result, run_span, trace_id_hex

        # A run-level span parents the whole agent trace, so each run is one self-describing
        # trace (model + fit metrics + verdict on the root); its trace id is stamped into
        # result.json so the across-example aggregator can join usage back to it.
        span_attrs = {
            "scenario": scenario.name, "example_version": scenario.version,
            "model": args.model, "driver": args.driver, "temperature": args.temperature,
            "recursion_limit": args.recursion_limit, "n_starts": args.n_starts,
            "git_sha": git_sha, "run_started_at": run_started_at,
            # The text the agent was given is an INPUT to the run, like the code sha — so the
            # trace root carries both, and a Phoenix query can group or split traces on either.
            # Two numbers rather than one so a change is attributable to the prompts
            # (prompts.py) or the skills (_skills/), which are edited independently.
            "prompt_version": base_meta.get("prompt_version"),
            "skills_version": base_meta.get("skills_version"),
            # Which ARM, so a Phoenix query can split an A/B the same way the metrics CSV does.
            # Not derivable from prompt_version: an arm that changes no prompt text (see
            # petab_harness.ablations) leaves it identical across both halves.
            "ablation_arm": base_meta.get("ablation_arm"),
        }
        exit_code, result = 1, {}
        with run_span(tracing, name="model_build_run", attributes=span_attrs) as span:
            base_meta = {**base_meta,
                         "phoenix_project": tracing.project_name if tracing else None,
                         "phoenix_trace_id": trace_id_hex(span)}
            # Publish the live marker now that the trace id is known, so the explorer can
            # watch this run's turns + model variants stream in before it finishes.
            _write_live_marker(work, base_meta, scenario)
            try:
                # The biocurator roster/guidance is spliced into the lead prompt INSIDE
                # run_deepagents, under the SAME guard that actually wires the subagent, so the
                # prompt can't promise a biocurator that wasn't added. Its tools are the
                # curated agent.DEFAULT_BIO_TOOLS allowlist (+ literature under the
                # `bio_literature` ablation, PETAB_BIO_LITERATURE=1).
                agent_out = await run_deepagents(
                    session,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=api_key,
                    system_prompt=scenario.system_prompt,
                    task=scenario.agent_task,
                    recursion_limit=args.recursion_limit,
                    temperature=args.temperature,
                    n_starts=args.n_starts,
                    max_wall_seconds=args.max_wall_seconds,
                    progress_probe=raw_call_tool,
                    bio_sessions=bio_sessions,
                    bio_literature=ablations.enabled("bio_literature"),
                    # Hand-curated biological entities for the blinded biocurator (issue #91);
                    # empty for scenarios with no lookupable biology (the abstract toys).
                    curated_entities=getattr(scenario, "biocurator_entities", None),
                )
                stop_reason = agent_out.get("stop_reason", "completed")
                if stop_reason != "completed":
                    print(f"[agent] stopped early: {stop_reason} "
                          f"(raise --recursion-limit if it ran out of steps)")
                # Repeated-tool-error circuit breaker (agent.py): the number of failed tool
                # calls this run, and any tool it disabled after N consecutive identical errors.
                meta = {**base_meta, "stop_reason": stop_reason,
                        "n_tool_errors": agent_out.get("n_tool_errors", 0),
                        "disabled_tools": agent_out.get("disabled_tools", []),
                        # make_changes is the only editor, so its call-level fail rate is the
                        # run's most direct health signal. Client-side, so it counts the attempts
                        # `edit_rejects` cannot see (argument-shape failures that never reached
                        # the server, and breaker short-circuits).
                        "make_changes_calls": agent_out.get("make_changes_calls", {}),
                        # What was actually SENT for this model: the provider pin and the
                        # reasoning block. Recorded because the package is editable-installed
                        # from a shared checkout, so a config change lands on every task that
                        # starts after it -- a run must say what it ran with rather than leaving
                        # it to be inferred from submission time.
                        "openrouter_extra_body": agent_out.get("openrouter_extra_body", {}),
                        # Which bio tools the biocurator actually got (empty = a "dumb" run — the
                        # biocurator ran on the model's own knowledge, no bio tools) — recorded to
                        # result.json so a sweep can see the biocurator's status.
                        "biocurator_tools": agent_out.get("biocurator_tools", []),
                        # The entity names auto-injected into the biocurator (observables + any
                        # model species), and any `task(...)` calls bounced for carrying kwargs the
                        # tool drops (a `prompt=` — issue #91). Recorded so a sweep can confirm the
                        # biocurator was actually handed its entities and catch a relapse.
                        "biocurator_entities": agent_out.get("biocurator_entities", []),
                        "unexpected_task_kwargs": agent_out.get("unexpected_task_kwargs", []),
                        # Subagent `task` calls whose returned report was empty/whitespace (the
                        # analyst's final completion was blank, so deepagents would have forwarded
                        # "" to the lead). The guard re-dispatches once (`_retries`) and, if still
                        # blank, substitutes a visible marker (`_markers`) — recorded so a sweep can
                        # see discarded analyst work rather than silently losing it.
                        "empty_subagent_reports": agent_out.get("empty_subagent_reports", 0),
                        "empty_subagent_retries": agent_out.get("empty_subagent_retries", 0),
                        "empty_subagent_markers": agent_out.get("empty_subagent_markers", 0),
                        # The panel the lead could delegate to, and whether deepagents' uncurated
                        # `general-purpose` subagent was confirmed off (false = confirmed off;
                        # true/null = contaminated or unverified — issue #112). Recorded so a
                        # sweep can filter such runs out instead of re-reading logs.
                        "subagent_names": agent_out.get("subagent_names", []),
                        "general_purpose_subagent": agent_out.get("general_purpose_subagent"),
                        # Provenance of the literature the biocurator READ (per call: query +
                        # returned DOIs/PMIDs; only populated under the `bio_literature` ablation) — an audit
                        # trail for analysis and for the leakage check on published problems.
                        "biocurator_literature": agent_out.get("biocurator_literature", []),
                        # Whole-run token usage (lead + subagents) summed in-process by
                        # agent._token_accountant: prompt / completion / cache_read / total.
                        # cache_read is the cached-prompt count, so an offline cost estimate can
                        # be cache-aware without re-deriving anything from the Phoenix span DB.
                        "token_usage": agent_out.get("token_usage")}
                exit_code, result = await _judge(scenario, call, baseline, work, meta, clock, t0)
                record_run_result(span, result)
            except BaseException as exc:
                # A provider or MCP-connection stall (httpx ReadTimeout/ConnectError, often
                # wrapped in an anyio TaskGroup BaseExceptionGroup whose leaf is a background
                # CancelledError) can escape the agent loop's own guard, OR kill the session
                # mid-judge. Don't let it vanish the run with rc=1 and NO result.json — which
                # strands the batch and the cross-run aggregator. Write a degraded result and
                # exit cleanly. (_should_reraise lets a real interrupt / top-level cancellation
                # through untouched.)
                if _should_reraise(exc):
                    raise
                rc = server_proc.poll() if server_proc is not None else None
                result = _write_degraded_result(work, base_meta, scenario, _summarize_exc(exc),
                                                server_returncode=rc, clock=clock)
                exit_code = 1
                with contextlib.suppress(Exception):
                    record_run_result(span, result)  # span still live inside the `with`
        return exit_code, result
    except BaseException as exc:
        # Same guarantee for the window the inner guard can't see: preflight, baseline
        # scoring, the scripted judge, and the deepagents setup before the run span opens.
        # There is no live span here (the failure is before/outside it), so just persist
        # result.json. A missing API key raises SystemExit and is re-raised, not degraded.
        if _should_reraise(exc):
            raise
        rc = server_proc.poll() if server_proc is not None else None
        return 1, _write_degraded_result(work, base_meta, scenario, _summarize_exc(exc),
                                         server_returncode=rc)
    finally:
        if tracing is not None:
            tracing.flush()  # export any buffered spans before we (maybe) exit
            print(f"[phoenix] traces recorded under project '{tracing.project_name}' "
                  f"-> {tracing.ui_url}")
            if args.phoenix_keep_alive:
                print("[phoenix] keeping the process alive so the traces stay browsable; "
                      "press Ctrl-C to exit.")
                try:
                    while True:
                        await asyncio.sleep(3600)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n[phoenix] shutting down.")
