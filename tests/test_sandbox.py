"""The bubblewrap sandbox an agent's shell runs in, and the deepagents backend around it.

The sandbox is the only thing standing between an agent's shell and a shared node: the operator's
credentials in ``$HOME``, the harness source, and the answer to the problem sitting one directory
above the workspace. So these tests do not check that the sandbox works — they check what it
must not reach, that the environment is built rather than inherited, that a timeout kills the
command and not merely ``bwrap``, and that a node without bubblewrap is refused instead of run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemot.agent import sandbox as S  # noqa: E402

#: Skipped where there is no bubblewrap (macOS: the sandbox is Linux-only by design), but never
#: under CI: there a missing bwrap fails `sandbox_preflight` and a broken one fails it with bwrap's
#: own error. Every isolation check below skipped green in CI until ci.yml installed bubblewrap.
requires_bwrap = pytest.mark.skipif(
    S._bwrap_binary() is None and not os.environ.get("CI"),
    reason="bubblewrap not installed on this node",
)


def _sandbox(tmp_path: Path, timeout: float = 60.0):
    pytest.importorskip("deepagents")
    work = tmp_path / "ex" / "work"
    work.mkdir(parents=True)
    (tmp_path / "ex" / "run.py").write_text("TRUE_ANT = 'secret'\n")
    S.sandbox_preflight(work, n_cpus=2)
    audit = S.new_shell_audit()
    backend = S.sandbox_backend_class()(
        work, env_prefix=S.default_env_prefix(), n_cpus=2, timeout=timeout, audit=audit
    )
    return work, backend, audit


@requires_bwrap
def test_preflight_proves_the_sandbox_and_makes_its_dirs(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    out = S.sandbox_preflight(work, n_cpus=4)
    assert out["bwrap"].startswith("bubblewrap")
    assert (work / S.SANDBOX_HOME_DIR).is_dir() and (work / S.SANDBOX_TMP_DIR).is_dir()


def test_preflight_refuses_without_bwrap(tmp_path, monkeypatch):
    """No unsandboxed fallback: a node without bubblewrap does not run an agent shell."""
    monkeypatch.setattr(S, "_bwrap_binary", lambda: None)
    with pytest.raises(RuntimeError, match="no unsandboxed fallback"):
        S.sandbox_preflight(tmp_path, n_cpus=1)


@requires_bwrap
def test_shell_sees_the_workspace_and_env_and_nothing_else(tmp_path):
    """The example directory above the workspace holds the answer; the repo holds the harness;
    $HOME holds credentials. None of them may exist inside the sandbox."""
    import os

    work, backend, _ = _sandbox(tmp_path)
    (work / "hello.txt").write_text("hi\n")
    assert "hi" in backend.execute("cat hello.txt && pwd").output
    assert "/work" in backend.execute("pwd").output
    # the parent of the workspace is NOT the example dir: `..` is the sandbox root
    assert backend.execute("cat ../run.py").exit_code != 0
    assert backend.execute(f"test -e {tmp_path / 'ex' / 'run.py'}").exit_code != 0
    home = os.path.expanduser("~")
    assert backend.execute(f"test -e {home!r}").exit_code != 0
    repo_src = Path(__file__).parents[1] / "src"
    assert backend.execute(f"test -e {repo_src}").exit_code != 0
    # the harness itself (an editable install pointing at src/) is therefore not importable...
    assert backend.execute("python -c 'import gemot'").exit_code != 0
    # ...while the scientific stack is
    assert (
        backend.execute("python -c 'import amici, pypesto, petab, scipy, sklearn'").exit_code == 0
    )
    # and the environment is read-only
    assert backend.execute(f"touch {S.default_env_prefix()}/bin/_probe").exit_code != 0


@requires_bwrap
def test_shell_has_no_network(tmp_path):
    _, backend, _ = _sandbox(tmp_path)
    res = backend.execute(
        "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\""
    )
    assert res.exit_code != 0


@requires_bwrap
def test_shell_env_is_built_from_an_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
    monkeypatch.setenv("WANDB_API_KEY", "w-secret-value")
    _, backend, _ = _sandbox(tmp_path)
    out = backend.execute("env").output
    assert "secret-value" not in out and "OPENROUTER_API_KEY" not in out
    assert "PETAB_MCP_FIT_N_WORKERS=2" in out and "OMP_NUM_THREADS=2" in out
    assert "HOME=/work/.home" in out and "TMPDIR=/tmp" in out


@requires_bwrap
def test_shell_audit_counts_commands_failures_and_timeouts(tmp_path):
    work, backend, audit = _sandbox(tmp_path, timeout=60.0)
    assert backend.execute("echo ok").exit_code == 0
    assert backend.execute("exit 3").exit_code == 3
    import time

    t0 = time.monotonic()
    res = backend.execute("sleep 30", timeout=1)
    # ...and the kill reaches the sandboxed command itself: a killed bwrap whose command lived
    # on would hold the output pipe and block here for the full 30s.
    assert time.monotonic() - t0 < 10
    assert res.exit_code == 124 and "timed out" in res.output
    assert audit["n_shell_commands"] == 3
    assert audit["n_nonzero_exits"] == 2 and audit["n_timeouts"] == 1
    assert audit["commands"] == ["echo ok", "exit 3", "sleep 30"]
    assert audit["shell_seconds"] > 0
    # writes land in the real workspace, where the judge will read them
    backend.execute("echo delivered > out.txt")
    assert (work / "out.txt").read_text().strip() == "delivered"


def test_file_tools_are_confined_to_the_mounted_workspace(tmp_path):
    """The file tools see the workspace as `/`; `..` and absolute paths outside it are refused by
    deepagents' virtual mode."""
    pytest.importorskip("deepagents")
    example_dir = tmp_path / "ex"
    work = example_dir / "work"
    work.mkdir(parents=True)
    (example_dir / "run.py").write_text("TRUE_ANT = 'secret'\n")
    (work / "problem.yaml").write_text("format_version: 1\n")
    backend = S.sandbox_backend_class()(
        work, env_prefix=S.default_env_prefix(), n_cpus=1, timeout=10, audit=S.new_shell_audit()
    )
    ok = backend.read("/problem.yaml")
    assert ok.error is None and "format_version" in str(ok.file_data)
    for escape in ("/../run.py", str(example_dir / "run.py")):
        # virtual mode refuses traversal by RAISING (deepagents FilesystemBackend._resolve_path);
        # an error result without the file's contents would be equally acceptable.
        try:
            out = backend.read(escape)
        except ValueError:
            continue
        assert out.file_data is None or "secret" not in str(out.file_data), escape
        assert out.error, escape
