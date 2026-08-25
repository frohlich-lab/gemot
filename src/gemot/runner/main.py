"""The pipeline: clean, start the server, open the sessions, drive, archive, tear down."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ..bio_sessions import open_bio_sessions
from ..explorer.archive import archive_run
from ..fit_budget import (FIT_BUDGET_CEILING_S, FIT_N_WORKERS, FIT_PER_START_TIMEOUT_S,
                          fit_time_budget_for)
from .drive import _drive
from .endpoints import BIO_MCP_URLS
from .failure import _should_reraise, _summarize_exc, _write_degraded_result
from .provenance import _base_meta, _now_iso
from ..scenario import Scenario
from .server import _SERVER_BOOTSTRAP, _free_port, _wait_until_listening

# --------------------------------------------------------------------------- steps
def cleanup(scenario: Scenario) -> Path:
    """Recreate work/ from the pristine broken problem (also drops old provenance)."""
    work = scenario.example_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(scenario.problem_dir, work)
    print(f"[cleanup] fresh working copy at {work.relative_to(scenario.example_dir.parent)}")
    return work


# --------------------------------------------------------------------------- main
async def _main_async(scenario: Scenario, args: argparse.Namespace) -> "tuple[int, dict]":
    work = cleanup(scenario)

    # Capture the true run start once, before the server launch, so every degraded-result path
    # (including a teardown-time abort in this function) reports the real start instead of
    # re-stamping it at abort time.
    started_at = _now_iso()

    work_yaml = work / "problem.yaml"
    # AMICI builds are keyed by SBML hash and never go stale, so keep them OUTSIDE work/
    # (wiped each run) to avoid recompiling identical models on repeat runs.
    build_root = scenario.example_dir / ".amici_build"
    host, port = "127.0.0.1", _free_port()
    server_cmd = [
        sys.executable, "-c", _SERVER_BOOTSTRAP,
        str(work_yaml), str(build_root), args.reference_validation, host, str(port),
        str(args.n_starts),
    ]
    server_log_path = work / "server.log"
    amici_log_path = work / "amici_build.log"
    server_log = server_log_path.open("w")
    amici_log = amici_log_path.open("w")

    print(f"[server] launching on http://{host}:{port}/mcp")
    print(f"[server] server log -> {server_log_path.name} | AMICI build log -> {amici_log_path.name}")
    # Split the streams: fd1 (stdout) carries the C++ build toolchain output that would
    # otherwise corrupt a stdio channel -> amici_build.log; fd2 (stderr) carries uvicorn
    # request logs + Python/AMICI logging -> server.log.
    # Export the harness's resolved Fides per-start cap + worker count onto the server so its
    # candidate fits use the SAME cap AND parallelism the harness sized its sse_read_timeout +
    # stall watchdog to (fit_budget.py) — otherwise the budget would not match the real wall-clock.
    # PYTHONFAULTHANDLER: on a fatal signal (SIGSEGV/SIGBUS/SIGFPE/SIGABRT) the interpreter dumps
    # a C-level traceback of EVERY thread to fd2 — which is server.log, already archived per task.
    # Added after five runs of the 2026-08-14 draft-arm sweep died with the server killed by
    # signal 11: the logs ended mid-AMICI parameter-mapping with no indication of where, because a
    # segfaulting process writes nothing on its way out. A core dump would say more, but cores are
    # disabled by default on the compute nodes (ulimit -c 0) and core_pattern pipes to
    # systemd-coredump, which stores node-locally and is not readable by the job's user — so the
    # in-process traceback is the artifact that actually survives. array_task.sh raises the core
    # limit too, so a core lands as well where the node permits it.
    server_env = {**os.environ,
                  "PYTHONFAULTHANDLER": "1",
                  "PETAB_MCP_FIT_PER_START_TIMEOUT": str(FIT_PER_START_TIMEOUT_S),
                  "PETAB_MCP_FIT_N_WORKERS": str(FIT_N_WORKERS)}
    try:
        proc = subprocess.Popen(server_cmd, stdout=amici_log, stderr=server_log, env=server_env)
    except Exception:
        for handle in (server_log, amici_log):
            with contextlib.suppress(Exception):
                handle.close()
        raise
    try:
        _wait_until_listening(host, port, proc, server_log_path)
        url = f"http://{host}:{port}/mcp"
        # A single make_changes commit fits the model with an n_starts multistart that streams
        # nothing back for its whole duration; the client's default sse_read_timeout (300s) would
        # drop that response and hang the call.
        #
        # This is the CEILING, not the fit budget. mcp layers two independent timeouts and they
        # do NOT compose the way the names suggest:
        #
        #   * sse_read_timeout becomes httpx's read timeout on the client (mcp/client/
        #     streamable_http.py: `httpx.Timeout(timeout, read=sse_read_timeout)`). Per SESSION,
        #     so it cannot know any call's n_starts.
        #   * ClientSession's read_timeout_seconds is an `anyio.fail_after` in the session layer,
        #     and it takes precedence PER CALL (mcp/shared/session.py) -- but only ever to cut a
        #     call SHORTER. It cannot lift the httpx ceiling above.
        #
        # So the exact per-call budget has to be applied downward from a ceiling that is high
        # enough for the largest fit the run could ask for -- and that number is unknowable here:
        # make_changes takes an n_starts that REPLACES the default, the agent chooses it at
        # runtime, and scenario.build_true_model floors it higher still (bruno 400, fujita 300).
        # This used to be `fit_time_budget(2 * n_starts)`; the 2x was a guess that still sat
        # below the scenario floors on the scripted path.
        #
        # Hence: make this ceiling deliberately NON-BINDING and let clock._wrap_call_timeout do
        # the real enforcement, exactly, off each call's own n_starts.
        #
        # Strictly ABOVE FIT_BUDGET_CEILING_S, which is what fit_time_budget_for clamps every
        # per-call budget to. Setting the two equal would leave the worst case (a hallucinated
        # n_starts that clamps to the ceiling) as a race between an anyio timer and httpx's read
        # deadline -- and if httpx won, the connection would be torn down instead of the call
        # returning a clean 408, which is the abort path this whole redesign exists to avoid.
        # The margin makes the per-call cut PROVABLY the tighter one.
        #
        # A flat constant, deliberately NOT args.max_wall_seconds. Tying it to the run budget
        # reads tidy -- no call can usefully outlive the run -- but it hands this timeout a way
        # to bind again: a fit that runs past a short --max-wall-seconds would surface as an
        # httpx read timeout instead of the clean "wall_time_budget_exceeded" stop the driver is
        # there to produce, i.e. a confusing error in place of a scored result. The wall budget
        # is already enforced where it can stop cleanly (agent.driver, at a step boundary).
        async with contextlib.AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(
                    url,
                    # Above the CEILING *and* above this run's own default budget: a 1-worker
                    # run with a large --n-starts can compute a default budget past the ceiling
                    # (the ceiling bounds what the MODEL may ask for, not what the operator
                    # configured), and this has to stay looser than both.
                    sse_read_timeout=max(FIT_BUDGET_CEILING_S,
                                         fit_time_budget_for(None, args.n_starts)) * 1.2)
            )
            # Session-wide fallback UNDER the per-call cut. mcp prefers a per-request
            # read_timeout_seconds outright (it does not take a min), so clock._wrap_call_timeout
            # still overrides this on every tool call. It matters in the two windows that shadow
            # does not cover: `initialize()` below, which would otherwise run against the
            # non-binding ceiling, and the case where installing the shadow FAILS -- which
            # previously left the run with no effective call timeout at all.
            session = await stack.enter_async_context(ClientSession(
                read, write,
                read_timeout_seconds=timedelta(
                    seconds=fit_time_budget_for(None, args.n_starts))))
            await session.initialize()
            # SECONDARY MCP sessions: the blinded bio-knowledge server(s) for the biocurator, on
            # the deepagents panel every run. The endpoints are HARDCODED (BIO_MCP_URLS:
            # BioContextAI + OmniPath) — no custom-endpoint config. Each is opened in its OWN
            # SUPERVISED, ISOLATED task by open_bio_sessions, kept separate from the petab session
            # (which the _ToolClock shadows) and from the run's fatal scope: a transport failure on
            # a bio session — unreachable, a slow/hung connect, or a mid-run ClosedResourceError —
            # is contained and degrades the biocurator to "dumb", NEVER aborting the run (issue
            # #86). LLM-driver only (empty for scripted). The biocurator is added REGARDLESS — if
            # no bio tools resolve it runs "dumb" on the model's own knowledge (biocurator_tools).
            bio_urls = BIO_MCP_URLS if args.driver != "scripted" else ()
            async with open_bio_sessions(bio_urls) as bio_sessions:
                exit_code, result = await _drive(scenario, session, work, args,
                                                 run_started_at=started_at, server_proc=proc,
                                                 bio_sessions=bio_sessions)
                # Snapshot the run's on-disk half (result.json + version store) into the runs
                # archive while work/ still exists — the next run of this example wipes it. The
                # run explorer (examples/explore_ui.py) stitches this with the persisted Phoenix
                # trace. Best-effort: archive_run never raises.
                if result:
                    # Record WHERE it went. The archive is keyed by run id, but every sweep and
                    # recover_run.py key by <job>_<task>; without this the history exists and is
                    # unfindable from a result, which is what made a bash copier look necessary.
                    _dest = archive_run(work, result)
                    if _dest is not None:
                        result["archive_dir"] = str(_dest)
                        with contextlib.suppress(Exception):
                            (work / "result.json").write_text(json.dumps(result, indent=2))
                return exit_code, result
    except BaseException as exc:
        # `_drive`'s own guard covers a stall/provider failure inside the agent loop or the
        # judge. This one covers the SAME class of failure (an anyio BaseExceptionGroup whose
        # leaf is e.g. httpx.ReadTimeout) escaping from OUTSIDE that scope instead — closing the
        # streamablehttp_client/ClientSession connection to an already-wedged server after
        # `_drive` already returned, or `_wait_until_listening` itself. Without this, such a
        # failure sails past `_drive`'s guard untouched (this call site is the one place between
        # `_drive` returning and `run_scenario`'s outermost catch), and the outermost catch does
        # NOT write a degraded result (it assumes an inner layer already did) — so the run
        # vanishes with exit code 1 and no result.json, stranding the batch/aggregator. Mirrors
        # `_drive`'s guard exactly; `_should_reraise` still lets a real interrupt through.
        if _should_reraise(exc):
            raise
        # ``proc`` is always bound here: this guard's try-block starts only after the server
        # subprocess was launched (its own launch failure re-raises separately above).
        rc = proc.poll()
        result = _write_degraded_result(work, _base_meta(scenario, args, started_at),
                                        scenario, _summarize_exc(exc), server_returncode=rc)
        return 1, result
    finally:
        # Tear down without masking any in-flight exception.
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=10)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=5)
        for handle in (server_log, amici_log):
            with contextlib.suppress(Exception):
                handle.close()
