"""The pipeline: clean, start the server, open the sessions, drive, archive, tear down."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ..bio_sessions import open_bio_sessions
from ..explorer.archive import archive_run
from ..fit_budget import FIT_N_WORKERS, FIT_PER_START_TIMEOUT_S, fit_time_budget
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
    server_env = {**os.environ,
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
        # A single make_changes commit fits the model with an n_starts multistart that streams nothing
        # back for its whole duration; the client's default sse_read_timeout (300s) would drop that
        # response and hang the call. Size it to the worst-case fit (n_starts * Fides per-start cap
        # + compile margin), the SAME budget the stall watchdog uses, so a slow-but-healthy fit
        # always returns instead of being mistaken for a hang (see _harness/fit_budget.py).
        async with contextlib.AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(url, sse_read_timeout=fit_time_budget(args.n_starts))
            )
            session = await stack.enter_async_context(ClientSession(read, write))
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
                    archive_run(work, result)
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
