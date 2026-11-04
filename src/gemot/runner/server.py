"""Launching the petab-mcp server the run drives, as a subprocess.

HTTP rather than stdio because AMICI's C++ build writes to the server's stdout while compiling,
which would corrupt a stdio JSON-RPC channel. The bootstrap carries a parent-death watchdog so a
SIGKILLed run.py cannot leak a uvicorn server.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from ..fit_budget import FIT_N_WORKERS, FIT_PER_START_TIMEOUT_S
from ..scenario import Scenario

# --------------------------------------------------------------------------- server
# Launched with:
#   python -c <bootstrap> <yaml> <build_root> <ref_mode> <host> <port> <n_starts> <validation_tsv>
#             <score_shipped> <entity_map> <constraints>
# <validation_tsv> is "" for an example with no held-out data, <score_shipped> is "1" to compile
# and score a problem's SHIPPED model at its nominal values before listening
# (create_server(score_shipped_model=True)) and "0" for a zero-shot run, which has nothing to
# score yet. Every argument is always passed — positional and fixed-arity, so a missing one is a
# hard ValueError ("not enough values to unpack") at startup rather than a server that silently
# comes up without the judge's held-out read registered. <entity_map> is "" off the
# indra_statements arm; on it, the server offers make_changes the `statements` route alone, which
# a map implies (frohlich-lab/PEtab-MCP#169), documented in the server's own description.
# <constraints> is "" off the causal_constraints arm; on it, the problem's constraints file, which
# the server checks every change against (create_server(constraints=...)). Updates stay off
# (constraint_updates="off"), so the server registers no `set_constraints` and no agent can change
# the claims it is held to; an arm in which the agent may is a decision of its own.
_SERVER_BOOTSTRAP = (
    "import sys, os, threading, time\n"
    # Parent-death watchdog: if run.py dies (incl. SIGKILL by the batch watchdog, which skips
    # run.py's own `finally` teardown), we'd otherwise leak this uvicorn server forever. Exit
    # within ~5s of being orphaned (getppid changes when the parent dies -> reparented). Portable
    # (macOS/Linux), unlike PR_SET_PDEATHSIG.
    "_ppid = os.getppid()\n"
    "def _exit_if_orphaned():\n"
    "    while True:\n"
    "        time.sleep(5)\n"
    "        if os.getppid() != _ppid:\n"
    "            os._exit(0)\n"
    "threading.Thread(target=_exit_if_orphaned, daemon=True).start()\n"
    "import uvicorn\n"
    "_Config = uvicorn.Config\n"
    # Silence uvicorn's per-request access log so the server's stdout carries only
    # AMICI's C++ build-toolchain output (captured to work/amici_build.log).
    "uvicorn.Config = lambda *a, **k: _Config(*a, **{**k, 'access_log': False})\n"
    "from petab_mcp.server import create_server\n"
    "yaml_path, build_root, ref_mode, host, port, n_starts, validation_tsv, score_shipped, \\\n"
    "    entity_map, constraints = sys.argv[1:11]\n"
    # The agent's own commits don't pass n_starts, so make the server's
    # default multistart count match --n-starts — otherwise agent fits use the low library
    # default (5), too few for the harder several-parameter oscillator fits.
    "srv = create_server(yaml_path, build_root=build_root, reference_mode=ref_mode,\n"
    "                    auto_fit_starts=int(n_starts),\n"
    "                    validation_measurements=validation_tsv or None,\n"
    "                    score_shipped_model=(score_shipped == '1'),\n"
    "                    entity_map=entity_map or None,\n"
    "                    constraints=constraints or None,\n"
    "                    constraint_updates='off',\n"
    # A sweep's fit pool, by its scheduler's address (`petab-mcp-pool`, PEtab-MCP#225), read
    # under the name the server's own CLI reads. Unset, the fits run on this server's own worker
    # processes. Read off the environment the run records it from (`provenance._fit_settings`).
    "                    cluster=os.environ.get('PETAB_MCP_FIT_POOL') or None)\n"
    # fastmcp >=3.4 dropped `srv.settings`; host/port are passed to run() (forwarded to the
    # streamable-http transport's run_http_async(host=, port=)).
    "srv.run(transport='streamable-http', host=host, port=int(port))\n"
)


def _server_env(scenario: Scenario) -> dict[str, str]:
    """The environment the server is launched with, and so the one its fit settings are read
    from: the run records them by resolving this same dict (``provenance._fit_settings``).

    It exports the harness's resolved Fides per-start cap + worker count, so the server's candidate
    fits use the SAME cap AND parallelism the harness sized its read timeout + stall watchdog
    to (fit_budget.py); otherwise the budget would not match the real wall-clock. The problem's own
    AMICI tolerances go last, so they win over any inherited from the caller: they are part of what
    the example IS, and its reference bar was fitted under them.

    PYTHONFAULTHANDLER: on a fatal signal (SIGSEGV/SIGBUS/SIGFPE/SIGABRT) the interpreter dumps a
    C-level traceback of EVERY thread to fd2, which is server.log, already archived per task. Added
    after five runs of the 2026-08-14 draft-arm sweep died with the server killed by signal 11: the
    logs ended mid-AMICI parameter-mapping with no indication of where, because a segfaulting
    process writes nothing on its way out. A core dump would say more, but cores are disabled by
    default on the compute nodes (ulimit -c 0) and core_pattern pipes to systemd-coredump, which
    stores node-locally and is not readable by the job's user, so the in-process traceback is the
    artifact that actually survives. array_task.sh raises the core limit too, so a core lands as
    well where the node permits it.
    """
    return {
        **os.environ,
        "PYTHONFAULTHANDLER": "1",
        "PETAB_MCP_FIT_PER_START_TIMEOUT": str(FIT_PER_START_TIMEOUT_S),
        "PETAB_MCP_FIT_N_WORKERS": str(FIT_N_WORKERS),
        **scenario.amici_tolerance_env(),
    }


def _build_root(scenario: Scenario) -> Path:
    """Where the run's server builds its models: ``PETAB_MCP_BUILD_ROOT``, the name the server's
    own CLI reads, else the example's ``.amici_build``. Either lies outside ``work/``, which every
    run wipes, and builds are keyed by structure, so a model compiled once is reused.

    On a fit pool (``PETAB_MCP_FIT_POOL``) the pool's workers import every module the server
    builds, by the same absolute path. So the sweep task sets the pool's build root, on storage
    the workers share, and the runs of one model compile it once between them. gemot never clears
    it; the operator does. A pool without one is refused here, as ``petab-mcp --fit-pool`` refuses
    it: the example's own folder is private to the run's task, and a worker that found no module
    to import there would end every fit as ``infrastructure_error: fit_pool``.
    """
    root = os.environ.get("PETAB_MCP_BUILD_ROOT")
    if os.environ.get("PETAB_MCP_FIT_POOL") and not root:
        raise ValueError(
            "PETAB_MCP_FIT_POOL is set but PETAB_MCP_BUILD_ROOT is not: the pool's workers import "
            "the models this run's server builds, so the build root must be the pool's, on "
            "storage they share"
        )
    return Path(root) if root else scenario.example_dir / ".amici_build"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_listening(
    host: str, port: int, proc: subprocess.Popen, log: Path, timeout: float = 120.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode}); see {log}")
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"server did not start listening on {host}:{port} within {timeout}s")
