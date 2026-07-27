"""Launching the petab-mcp server the run drives, as a subprocess.

HTTP rather than stdio because AMICI's C++ build writes to the server's stdout while compiling,
which would corrupt a stdio JSON-RPC channel. The bootstrap carries a parent-death watchdog so a
SIGKILLed run.py cannot leak a uvicorn server.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

# --------------------------------------------------------------------------- server
# Launched with: python -c <bootstrap> <yaml> <build_root> <ref_mode> <host> <port> <n_starts>
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
    "yaml_path, build_root, ref_mode, host, port, n_starts = sys.argv[1:7]\n"
    # The agent's own commits don't pass n_starts, so make the server's
    # default multistart count match --n-starts — otherwise agent fits use the low library
    # default (5), too few for the harder several-parameter oscillator fits.
    "srv = create_server(yaml_path, build_root=build_root, reference_mode=ref_mode,\n"
    "                    auto_fit_starts=int(n_starts))\n"
    "srv.settings.host = host\n"
    "srv.settings.port = int(port)\n"
    "srv.run(transport='streamable-http')\n"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_listening(host: str, port: int, proc: subprocess.Popen, log: Path,
                          timeout: float = 120.0) -> None:
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
