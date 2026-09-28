"""A bubblewrap sandbox for an agent's shell, and the deepagents backend that runs commands in it.

An agent that can run shell commands on this cluster runs them on a SHARED node, next to the
operator's ``$HOME`` (credentials), the harness source, and — one directory above a run's
workspace — the reference model and its score. deepagents' own ``LocalShellBackend`` executes on
the host with none of that hidden, so it is not used here and there is no unsandboxed fallback:
:func:`sandbox_preflight` refuses the node instead.

What the sandbox gives a command: the workspace read-write at ``/work`` (its cwd), the Python
environment and the system trees read-only, no network, no ``$HOME``, an environment built from
an allowlist rather than inherited, and its own process group so a timeout kills the command and
not just ``bwrap``. :func:`sandbox_backend_class` wraps that as a deepagents backend whose file
tools see the same workspace, and records every command in a shell audit for provenance.

LINUX ONLY, deliberately: ``bwrap`` needs unprivileged user namespaces, the read-only binds name
FHS paths, and the compilers below are the conda ``x86_64-linux-gnu`` toolchain. Everything that
runs an agent shell runs on the cluster, so this module does not abstract over platforms -- on
anything else :func:`sandbox_preflight` raises, which is the intended outcome rather than a gap.

Reasons and the failure each rule prevents: ``docs/RATIONALE.md``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

#: Where the workspace is mounted INSIDE the sandbox — the shell's cwd. The file tools show the
#: same directory as ``/`` (virtual mode); the prompt states both.
SANDBOX_WORKDIR = "/work"

#: Hidden directories inside the workspace that become the sandbox's ``$HOME`` and ``/tmp``.
SANDBOX_HOME_DIR = ".home"
SANDBOX_TMP_DIR = ".tmp"

#: Host trees bound read-only into the sandbox; one missing on the host is skipped.
_SANDBOX_RO_DIRS = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")

#: Variables copied into the sandbox when set (compute knobs); nothing else is inherited.
_PASSTHROUGH_ENV = (
    "CC",
    "CXX",
    "AMICI_PARALLEL_COMPILE",
    "PETAB_MCP_FIT_N_WORKERS",
    "SLURM_CPUS_PER_TASK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

#: Bound on the preflight's own commands, which do nothing but prove the sandbox starts.
_PREFLIGHT_TIMEOUT_S = 60.0

#: How much of each shell command the audit keeps.
_COMMAND_RECORD_CHARS = 2000
#: Same ceiling deepagents' own shell backend applies to one command's combined output.
_MAX_OUTPUT_BYTES = 100_000


def default_env_prefix() -> Path:
    """The Python environment this process runs from — the one tree the agent may execute from."""
    return Path(sys.prefix).resolve()


def _bwrap_binary() -> str | None:
    """The environment's own ``bwrap`` if present, else the first on ``PATH``; ``None`` if neither."""
    own = default_env_prefix() / "bin" / "bwrap"
    if own.is_file() and os.access(own, os.X_OK):
        return str(own)
    return shutil.which("bwrap")


def _sandbox_env(env_prefix: Path, n_cpus: int) -> dict[str, str]:
    """The sandbox's whole environment, built from scratch (see ``_PASSTHROUGH_ENV``)."""
    env = {
        "PATH": f"{env_prefix}/bin:/usr/bin:/bin",
        "HOME": f"{SANDBOX_WORKDIR}/{SANDBOX_HOME_DIR}",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        # The conda toolchain by its linux-64 name: see the module docstring on why this module
        # is Linux-only rather than platform-agnostic.
        "CC": str(env_prefix / "bin" / "x86_64-conda-linux-gnu-cc"),
        "CXX": str(env_prefix / "bin" / "x86_64-conda-linux-gnu-c++"),
        "PETAB_MCP_FIT_N_WORKERS": str(n_cpus),
        "OMP_NUM_THREADS": str(n_cpus),
    }
    for name in _PASSTHROUGH_ENV:
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def bwrap_argv(
    workspace: Path, env_prefix: Path, sandbox_env: dict[str, str], command: str
) -> list[str]:
    """The full ``bwrap`` command line that runs ``command`` in the sandbox described above."""
    argv = [
        _bwrap_binary() or "bwrap",
        # No `--new-session`: the timeout kill relies on one process group (see _run_sandboxed).
        "--unshare-all",
        "--die-with-parent",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for d in _SANDBOX_RO_DIRS:
        if os.path.isdir(d):
            argv += ["--ro-bind", d, d]
    argv += ["--ro-bind", str(env_prefix), str(env_prefix)]  # at its own path, for shebangs
    argv += ["--bind", str(workspace), SANDBOX_WORKDIR]
    argv += ["--bind", str(workspace / SANDBOX_TMP_DIR), "/tmp"]
    argv += ["--chdir", SANDBOX_WORKDIR]
    for k, v in sandbox_env.items():
        argv += ["--setenv", k, v]
    argv += ["/bin/bash", "-c", command]
    return argv


def _run_sandboxed(
    workspace: Path, env_prefix: Path, sandbox_env: dict[str, str], command: str, timeout: float
) -> subprocess.CompletedProcess:
    """One command under bwrap, bounded by ``timeout``; raises ``subprocess.TimeoutExpired`` with
    the partial stdout when it overruns. The sandbox gets an empty host environment (``--setenv``
    only) and its own process group, which the timeout kills whole (see RATIONALE.md)."""
    proc = subprocess.Popen(  # noqa: S603 - argv list, no shell on the host side
        bwrap_argv(workspace, env_prefix, sandbox_env, command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        env={},
        cwd=str(workspace),
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        raise subprocess.TimeoutExpired(exc.cmd, timeout, output=out, stderr=err) from None
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def sandbox_preflight(workspace: Path, env_prefix: Path | None = None, *, n_cpus: int = 1) -> dict:
    """Prove the sandbox on this node, or raise ``RuntimeError``: ``bwrap`` exists, a sandboxed
    command runs, the workspace is at ``/work`` and the interpreter imports, and the home
    directory is not visible. Returns a record for ``result.json``. There is no fallback."""
    env_prefix = env_prefix or default_env_prefix()
    bwrap = _bwrap_binary()
    if bwrap is None:
        raise RuntimeError(
            "sandboxing an agent's shell needs bubblewrap (`bwrap`), and it is not on PATH. "
            "There is deliberately no unsandboxed fallback."
        )
    for d in (SANDBOX_HOME_DIR, SANDBOX_TMP_DIR):
        (workspace / d).mkdir(exist_ok=True)
    env = _sandbox_env(env_prefix, n_cpus)
    home = os.path.expanduser("~")
    probe = (
        "set -e; test -d /work && echo WORK_OK; "
        f"python -c 'import sys; print(\"PY_OK\", sys.version_info[0])'; "
        f"test -e {home!r} && echo HOME_VISIBLE || echo HOME_HIDDEN; "
        "echo ENV_COUNT $(env | wc -l)"
    )
    try:
        proc = _run_sandboxed(workspace, env_prefix, env, probe, _PREFLIGHT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"the sandbox preflight did not finish in {_PREFLIGHT_TIMEOUT_S:.0f}s"
        ) from exc
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "WORK_OK" not in out or "PY_OK" not in out:
        raise RuntimeError(
            "the sandbox does not start on this node (bwrap exit "
            f"{proc.returncode}): {out.strip()[-500:]!r}. Unprivileged user namespaces may be "
            "disabled here; an agent shell does not run without them."
        )
    if "HOME_HIDDEN" not in out:
        raise RuntimeError("the sandbox can see the home directory — refusing to run the agent.")
    version = subprocess.run(
        [bwrap, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "bwrap": version,
        "bwrap_path": bwrap,
        "verified": ["workspace", "python", "home_hidden"],
        "env_prefix": str(env_prefix),
        "ro_dirs": [d for d in _SANDBOX_RO_DIRS if os.path.isdir(d)],
    }


def new_shell_audit() -> dict[str, Any]:
    """A fresh shell-audit record, seeded so ``result.json`` carries explicit zeros."""
    return {
        "n_shell_commands": 0,
        "shell_seconds": 0.0,
        "n_nonzero_exits": 0,
        "n_timeouts": 0,
        "commands": [],  # in order, truncated to _COMMAND_RECORD_CHARS
    }


def sandbox_backend_class():
    """The stock ``FilesystemBackend`` (virtual mode) plus a bwrap-contained ``execute`` in
    deepagents' ``SandboxBackendProtocol`` shape; output formatted like its own shell backend."""
    from deepagents.backends import FilesystemBackend
    from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

    class _BwrapSandbox(FilesystemBackend, SandboxBackendProtocol):
        def __init__(
            self,
            workspace: Path,
            *,
            env_prefix: Path,
            n_cpus: int,
            timeout: float,
            audit: dict[str, Any],
        ) -> None:
            super().__init__(root_dir=str(workspace), virtual_mode=True)
            self._workspace = Path(workspace)
            self._env_prefix = env_prefix
            self._sandbox_env = _sandbox_env(env_prefix, n_cpus)
            self._default_timeout = float(timeout)
            self._audit = audit
            self._id = f"bwrap-{uuid.uuid4().hex[:8]}"

        @property
        def id(self) -> str:
            return self._id

        def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
            if not command or not isinstance(command, str):
                return ExecuteResponse(
                    output="Error: Command must be a non-empty string.",
                    exit_code=1,
                    truncated=False,
                )
            effective = float(timeout) if timeout else self._default_timeout
            audit = self._audit
            audit["n_shell_commands"] += 1
            audit["commands"].append(command[:_COMMAND_RECORD_CHARS])
            t0 = time.monotonic()
            try:
                proc = _run_sandboxed(
                    self._workspace, self._env_prefix, self._sandbox_env, command, effective
                )
            except subprocess.TimeoutExpired as exc:
                audit["n_timeouts"] += 1
                audit["n_nonzero_exits"] += 1
                partial = getattr(exc, "output", None) or getattr(exc, "stdout", None) or ""
                partial = (
                    partial.decode(errors="replace") if isinstance(partial, bytes) else partial
                )
                return ExecuteResponse(
                    output=(
                        f"{partial[-2000:]}\n\nError: command timed out after "
                        f"{effective:.0f}s and was killed. Run long jobs in the background "
                        "(`nohup ... > log 2>&1 &`) and poll the log."
                    ),
                    exit_code=124,
                    truncated=False,
                )
            finally:
                audit["shell_seconds"] += time.monotonic() - t0
            parts = []
            if proc.stdout:
                parts.append(proc.stdout)
            if proc.stderr:
                parts.extend(f"[stderr] {line}" for line in proc.stderr.strip().split("\n"))
            output = "\n".join(parts) if parts else "<no output>"
            truncated = False
            if len(output) > _MAX_OUTPUT_BYTES:
                output = (
                    output[:_MAX_OUTPUT_BYTES]
                    + f"\n\n... Output truncated at {_MAX_OUTPUT_BYTES} bytes."
                )
                truncated = True
            if proc.returncode != 0:
                audit["n_nonzero_exits"] += 1
                output = f"{output.rstrip()}\n\nExit code: {proc.returncode}"
            return ExecuteResponse(output=output, exit_code=proc.returncode, truncated=truncated)

    return _BwrapSandbox
