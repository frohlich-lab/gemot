"""One run, from a connected session to a written verdict.

Preflight the server, score the baseline, dispatch to the panel, a native CLI session or the
scenario's scripted canonical repair, then judge and archive. Wrapped so that anything
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

from .. import ablations, infrastructure
from .failure import _should_reraise, _summarize_exc, _write_degraded_result
from .judge import _judge
from .clock import _held_out_fits, _make_call, _wrap_call_timeout, _wrap_tool_clock
from .provenance import _agent_meta, _base_meta, _write_live_marker
from ..scenario import Baseline, Scenario


def _reasoning_effort(args) -> str | None:
    """The run's ``--reasoning-effort`` as sent: ``None`` for ``default``, keeping the CLI's
    default or the OpenRouter model's setting (:mod:`gemot.models`)."""
    effort = getattr(args, "reasoning_effort", None)
    return None if effort == "default" else effort


def _infrastructure_abort(exc: BaseException, logs: infrastructure.LogWatch) -> dict:
    """:func:`_write_degraded_result`'s keywords for an abort: the stop reason
    ``infrastructure_error: <cause>`` if it met an infrastructure failure, in its own text or the
    server's logs, and the log count and evidence that every deepagents result records."""
    new = logs(final=True)  # read to the end first, so the count is whole
    hit = infrastructure.find_in_error(exc) or new or logs.evidence
    return {
        "stop_reason": infrastructure.stop_reason(hit.cause) if hit is not None else None,
        "extra": {
            "infrastructure_log_lines": logs.lines,
            "infrastructure_evidence": hit.text if hit is not None else None,
        },
    }


async def _drive(
    scenario: Scenario,
    session: ClientSession,
    work: Path,
    args: argparse.Namespace,
    run_started_at: str | None = None,
    server_proc: "subprocess.Popen | None" = None,
    bio_sessions: "list | None" = None,
) -> "tuple[int, dict]":
    """Preflight, run the chosen driver, and judge — all on one live session.

    ``run_started_at`` (the real start captured by ``_main_async`` before the server launched)
    keeps a degraded result's start time truthful; ``server_proc`` lets an abort record whether
    the MCP server had crashed (see :func:`_write_degraded_result`).

    Returns ``(exit_code, result)`` (see :func:`_judge`).
    """
    t0 = time.monotonic()
    # Per-call read timeouts FIRST, so every later shadow (and `raw_call_tool` below) inherits
    # them: the session-wide sse_read_timeout can only know the run default, but n_starts is a
    # per-call argument. See _wrap_call_timeout.
    _wrap_call_timeout(session, args.n_starts, held_out_fits=_held_out_fits(scenario.example_dir))
    # Capture the un-timed call handle BEFORE the clock wrapper, so the agent's internal
    # early-stop polls (get_progress, once per super-step) don't inflate the tool clock /
    # tool_calls metric. "Un-timed" means unmeasured, not unbounded — it still carries the
    # per-call timeout applied above.
    raw_call_tool = session.call_tool
    clock = _wrap_tool_clock(session)  # split server/compute time from LLM time
    call = _make_call(session)
    from ..biology_blinding import AnonymousSession, read_names

    blind_names = read_names(work)
    agent_session = session
    if blind_names is not None:
        # Only harness preflight, the scripted self-test and judging use this caller.
        # run_deepagents keeps the actual session, which serves anonymous identifiers.
        call = blind_names.judge_call(call)
        agent_session = AnonymousSession(session, scenario.example_dir)
        # The finish gate also exposes diagnostics through its unmeasured reader.
        raw_call_tool = AnonymousSession(
            session, scenario.example_dir, call_tool=raw_call_tool
        ).call_tool

    # Stamp the run's start + provenance up front (before the slow baseline compile) so a
    # degraded result.json is still well-formed if preflight/baseline itself blows up.
    scripted = args.driver == "scripted"
    base_meta = _base_meta(scenario, args, run_started_at)
    run_started_at = base_meta["run_started_at"]
    git_sha = base_meta["git_sha"]

    # The two logs `runner.main` gives the server: a full disk shows there when nothing else says
    # so (gemot#39) -- an "ill-posed" build's real cause, a segfaulted server's last words.
    server_logs = infrastructure.LogWatch(work / "server.log", work / "amici_build.log")

    tracing = None
    try:
        desc = await call("describe_problem")
        print(
            f"[preflight] model '{desc.get('model_id')}' | "
            f"reactions={desc.get('n_reactions_detail')} | "
            f"observables={desc.get('observables')} | "
            f"free params={desc.get('free_parameters')}"
        )

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
            raw_bl = (
                baseline_fit.get("neg_log_likelihood") if isinstance(baseline_fit, dict) else None
            )
            try:
                bl_nll = float(raw_bl)
            except (TypeError, ValueError):
                bl_nll = float("nan")
            if not math.isfinite(bl_nll):
                bl_nll = None
            baseline = Baseline(nll=bl_nll, species=baseline_species, structure=baseline_structure)
            _bl_disp = (
                f"{baseline.nll:.6g}" if baseline.nll is not None else "n/a (first fit failed)"
            )
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

        # All model drivers share a root trace and independent judge. Native CLIs are observed
        # through their event streams and MCP gateway; only deepagents uses LangChain spans.
        cli_driver = args.driver in {"codex", "claude"}
        if args.phoenix:
            from ..tracing import setup_phoenix_tracing

            tracing = setup_phoenix_tracing(
                project_name=args.phoenix_project,
                **({"instrument_langchain": False} if cli_driver else {}),
            )

        if not cli_driver:
            api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise SystemExit(
                    "No OpenRouter API key: set OPENROUTER_API_KEY (or pass --api-key). "
                    "Get one at https://openrouter.ai/keys, or use --driver scripted for a "
                    "no-LLM self-test."
                )
        transport_label = {"codex": "Codex CLI", "claude": "Claude Code CLI"}.get(
            args.driver, "OpenRouter"
        )
        architecture = (
            "native session"
            if cli_driver
            else f"deepagents (recursion_limit={args.recursion_limit})"
        )
        print(f"[agent] driving {args.model} via {transport_label}, {architecture}")
        from ..tracing import record_run_result, run_span, trace_id_hex

        # A run-level span parents the whole agent trace, so each run is one self-describing
        # trace (model + fit metrics + verdict on the root); its trace id is stamped into
        # result.json so the across-example aggregator can join usage back to it.
        span_attrs = {
            "scenario": scenario.name,
            "example_version": scenario.version,
            "model": args.model,
            "driver": args.driver,
            "temperature": None if cli_driver else args.temperature,
            "reasoning_effort": _reasoning_effort(args),
            "recursion_limit": None if cli_driver else args.recursion_limit,
            "n_starts": args.n_starts,
            "git_sha": git_sha,
            "run_started_at": run_started_at,
            # The text the agent was given is an INPUT to the run, like the code sha — so the
            # trace root carries both, and a Phoenix query can group or split traces on either.
            # Two numbers rather than one so a change is attributable to the prompts
            # (prompts.py) or the skills (_skills/), which are edited independently.
            "prompt_version": base_meta.get("prompt_version"),
            "skills_version": base_meta.get("skills_version"),
            # Which ARM, so a Phoenix query can split an A/B the same way the metrics CSV does.
            # Not derivable from prompt_version: an arm that changes no prompt text (see
            # gemot.ablations) leaves it identical across both halves.
            "ablation_arm": base_meta.get("ablation_arm"),
            # ...and the default-INDEPENDENT twin, or the claim above is false across a default
            # flip: `ablation_arm` names only deviations, so "baseline" means different
            # configurations either side of 2026-08-15 (see ablations.state_key).
            "ablation_state": base_meta.get("ablation_state"),
        }
        exit_code, result = 1, {}
        with run_span(tracing, name="model_build_run", attributes=span_attrs) as span:
            base_meta = {
                **base_meta,
                "phoenix_project": tracing.project_name if tracing else None,
                "phoenix_trace_id": trace_id_hex(span),
            }
            # Publish the live marker now that the trace id is known, so the explorer can
            # watch this run's turns + model variants stream in before it finishes.
            _write_live_marker(work, base_meta, scenario)
            # What the guard below writes: `base_meta` until the driver returns, the driver's
            # whole report after.
            meta = base_meta
            try:
                if cli_driver:
                    from .. import prompts
                    from ..agent.native_cli import run_native_cli
                    from ..fit_budget import FIT_BUDGET_CEILING_S, fit_time_budget_for

                    timeout = getattr(args, f"{args.driver}_timeout", None)
                    if args.max_wall_seconds is not None:
                        remaining = args.max_wall_seconds - (time.monotonic() - t0)
                        timeout = min(timeout, remaining) if timeout is not None else remaining
                    if timeout is not None and timeout <= 0:
                        agent_out = {"stop_reason": "wall_time_budget_exceeded"}
                    else:
                        agent_out = await run_native_cli(
                            agent_session,
                            driver=args.driver,
                            model=args.model,
                            task=(
                                prompts.biology_blind_task(blind_names.protocols)
                                if blind_names is not None
                                else scenario.agent_task
                            ),
                            work=work,
                            executable=getattr(args, f"{args.driver}_executable", args.driver),
                            reasoning_effort=_reasoning_effort(args),
                            timeout=timeout,
                            tracing=tracing,
                            parent_span=span,
                            biology_blinded=blind_names is not None,
                            tool_timeout=max(
                                FIT_BUDGET_CEILING_S, fit_time_budget_for(None, args.n_starts)
                            )
                            * 1.2,
                        )
                else:
                    from ..agent import run_deepagents

                    # The default lead prompt and run_deepagents share BIOCURATOR_ENABLED,
                    # so the prompt cannot promise a curator that the selected arm omits.
                    agent_out = await run_deepagents(
                        agent_session,
                        model=args.model,
                        base_url=args.base_url,
                        api_key=api_key,
                        system_prompt=scenario.system_prompt,
                        task=scenario.agent_task,
                        recursion_limit=args.recursion_limit,
                        temperature=args.temperature,
                        reasoning_effort=_reasoning_effort(args),
                        n_starts=args.n_starts,
                        max_wall_seconds=args.max_wall_seconds,
                        progress_probe=raw_call_tool,
                        # The exact liveness signal, not a probe: Popen.poll() cannot confuse
                        # a server busy fitting with one that is gone (see _server_is_alive).
                        server_poll=(server_proc.poll if server_proc is not None else None),
                        infrastructure_probe=server_logs,
                        bio_sessions=bio_sessions,
                        bio_literature=ablations.enabled("bio_literature"),
                        # Hand-curated entities for the blinded biocurator (issue #91);
                        # empty for abstract scenarios with no lookupable biology.
                        curated_entities=scenario.agent_entities,
                        biocurator_model=getattr(args, "biocurator_model", None),
                        biology_blinded=blind_names is not None,
                        # The driver writes these anonymized protocols into the blind task.
                        input_protocols=blind_names.protocols if blind_names is not None else None,
                    )
                stop_reason = agent_out.get("stop_reason", "completed")
                # The logs once more, to their end: the driver stops reading them at its first
                # evidence, and a full disk can show in the run's last step.
                if (hit := server_logs(final=True)) is not None and not infrastructure.is_stop(
                    stop_reason
                ):
                    stop_reason = infrastructure.stop_reason(hit.cause)
                if stop_reason != "completed":
                    print(f"[agent] stopped early: {stop_reason}")
                # The driver's whole return, wholesale (see provenance._agent_meta), and the log
                # count for every run, so a contaminated one is visible even where nothing
                # stopped it.
                meta = {
                    **_agent_meta(base_meta, agent_out, stop_reason),
                    "infrastructure_log_lines": server_logs.lines,
                    "infrastructure_evidence": agent_out.get("infrastructure_evidence")
                    or (server_logs.evidence.text if server_logs.evidence else None),
                }
                # Is there a server left to judge against? Keyed on the PROCESS, not on
                # `stop_reason`: the driver only consults liveness on its retriable branch, so a
                # run whose server was OOM-killed can still arrive here labelled
                # wall_time_budget_exceeded, no_model_built, silent_stop or early_stop. Judging
                # then costs twelve reads x their full budget against something that is gone --
                # and in the wall-budget case that overspend is taken from the margin reserved
                # for scoring, which is how a task gets hard-killed with no result.json at all.
                rc = server_proc.poll() if server_proc is not None else None
                if infrastructure.is_stop(stop_reason):
                    # Not judged: a verdict on a run whose builds failed on a full disk scores
                    # the disk, and the judge's own compiles would meet it too; a wedged fit
                    # would hold every judge call behind it. Its own label, so the sweep can
                    # tell it apart and resubmit it.
                    print(
                        f"[run] {stop_reason} ({meta['infrastructure_evidence']}); "
                        "skipping the judge."
                    )
                    result = _write_degraded_result(
                        work,
                        meta,
                        scenario,
                        str(meta["infrastructure_evidence"]),
                        server_returncode=rc,
                        clock=clock,
                        stop_reason=stop_reason,
                    )
                    exit_code = 1
                elif rc is not None:
                    print(
                        f"[run] the MCP server process has exited ({rc}); skipping the judge "
                        "and writing what the run already produced."
                    )
                    # `meta`, NOT `base_meta`: an aborted run that DID work must keep its work.
                    # base_meta drops token_usage, n_tool_errors, make_changes_calls, the
                    # biocurator provenance and the subagent counters -- and without them
                    # `experiments.is_noop_infra_abort` reads the run as zero-work noise and
                    # DROPS it, which for bachmann 52859568_0 would have discarded 14 completed
                    # fits.
                    result = _write_degraded_result(
                        work,
                        meta,
                        scenario,
                        f"server_unreachable: MCP server exited ({rc})",
                        server_returncode=rc,
                        clock=clock,
                    )
                    exit_code = 1
                else:
                    exit_code, result = await _judge(
                        scenario, call, baseline, work, meta, clock, t0
                    )
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
                # `meta`, for the reason the server-died branch above gives: a judge that fails
                # after the driver returned must not cost the run its token_usage,
                # make_changes_calls and guard counters.
                result = _write_degraded_result(
                    work,
                    meta,
                    scenario,
                    _summarize_exc(exc),
                    server_returncode=rc,
                    clock=clock,
                    **_infrastructure_abort(exc, server_logs),
                )
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
        return 1, _write_degraded_result(
            work,
            base_meta,
            scenario,
            _summarize_exc(exc),
            server_returncode=rc,
            **_infrastructure_abort(exc, server_logs),
        )
    finally:
        if tracing is not None:
            tracing.flush()  # export any buffered spans before we (maybe) exit
            print(
                f"[phoenix] traces recorded under project '{tracing.project_name}' "
                f"-> {tracing.ui_url}"
            )
            if args.phoenix_keep_alive:
                print(
                    "[phoenix] keeping the process alive so the traces stay browsable; "
                    "press Ctrl-C to exit."
                )
                try:
                    while True:
                        await asyncio.sleep(3600)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    print("\n[phoenix] shutting down.")
