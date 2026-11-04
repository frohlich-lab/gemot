"""A run on a sweep's fit pool (gemot#105; frohlich-lab/PEtab-MCP#189, section "gemot").

With ``PETAB_MCP_FIT_POOL`` set, the run's own server joins the pool by address and its fits run
there, from models built under the pool's build root (``PETAB_MCP_BUILD_ROOT``). The run records
the pool in ``fit_settings`` and its wait for a worker as ``queue_seconds``. A pool the server
refuses ends the run as ``infrastructure_error: fit_pool``, which blames the pool, not the run.
Without the pool, nothing changes.

The pool runs in another process, as a sweep's does: ``petab_mcp.pool.main`` with a
``LocalCluster`` in place of its ``SLURMCluster``, as the server's own tests run it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from distributed import Client, Event

from gemot import infrastructure as I
from gemot.runner import main as M
from gemot.runner.cli import build_parser, run_scenario
from gemot.runner.clock import _make_call, _ToolClock
from gemot.runner.judge import _judge
from gemot.runner.provenance import _base_meta
from gemot.runner.server import _build_root
from gemot.scenario import Scenario

TOY = Path(__file__).parent / "data" / "toy_v1"

#: A pool whose workers are local processes, one per job, whose scheduler reports the versions of
#: this environment but for an ``amici`` of ``argv[1]`` (``-``: this environment's own).
_POOL = textwrap.dedent("""
    import sys
    from distributed import LocalCluster
    from petab_mcp import dask_engine, pool

    if __name__ == "__main__":
        if sys.argv[1] != "-":
            ours = dask_engine.versions
            dask_engine.versions = lambda: {**ours(), "amici": sys.argv[1]}
        pool._slurm_cluster = lambda port, min_jobs, max_jobs: LocalCluster(
            n_workers=max_jobs, threads_per_worker=1, scheduler_port=port, dashboard_address=":0"
        )
        raise SystemExit(pool.main(sys.argv[2:]))
""")

#: How long the pool's one worker stays busy once the run's fit is waiting for it.
_HELD_S = 4.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _pool(tmp_path: Path, workers: int = 1, amici: str = "-"):
    """A pool of ``workers`` in another process; yields its published address. Its process group
    is killed on exit."""
    (tmp_path / "pool.py").write_text(_POOL, encoding="utf-8")
    published = tmp_path / "address"
    flags = ["--slurm-max-jobs", str(workers), "--port", str(_free_port())]
    flags += ["--address-file", str(published)]
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "pool.py"), amici, *flags], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 300
        while not published.exists():
            assert proc.poll() is None and time.monotonic() < deadline, "the pool never published"
            time.sleep(0.2)
        yield published.read_text(encoding="utf-8").strip()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _scenario(tmp_path: Path) -> Scenario:
    problem = tmp_path / "toy" / "problem"
    shutil.copytree(TOY, problem)
    return Scenario("toy", problem, lambda *args: None)


class _Judged:
    """What the judge reads off a scenario besides its name and folder: a reference bar, which this
    test has no use for. The toy's n differs from it, so the run is not scored; its timing and
    settings are recorded all the same."""

    name = "toy"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, example_dir: Path) -> None:
        self.example_dir = example_dir

    def reference(self) -> dict:
        return {"nll": 0.0, "k": 2, "n": 1}


class _Baseline:
    nll = 100.0
    species = {"A", "B"}


def test_a_run_on_a_pool_fits_there_and_records_the_pool_and_its_wait(
    tmp_path, amici_build_root, monkeypatch
):
    """#105's first check. The server is launched by the run's own bootstrap, the toy is refitted
    through it, and the judge writes result.json. The pool's one worker is held from outside until
    the fit has waited for it, as another run's batch would hold it, so the wait is recorded as the
    pool's, not the run's own fit time."""
    scenario = _scenario(tmp_path)
    work = M.cleanup(scenario)
    args = build_parser(scenario).parse_args(["--driver", "scripted", "--n-starts", "2"])
    with _pool(tmp_path) as address, Client(address) as client:
        monkeypatch.setenv("PETAB_MCP_FIT_POOL", address)
        monkeypatch.setenv("PETAB_MCP_BUILD_ROOT", str(amici_build_root))
        release = Event("gemot-test-release", client=client)
        held = client.submit(release.wait, 600, pure=False)
        seen: list[str] = []

        def _release_once_the_fit_waits() -> None:
            # A fit's batches run `dask_engine._work`, so their keys show the fit reached the
            # pool's scheduler. Matched on the key, not its prefix: dask's prefix drops the `_`.
            try:
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    keys = client.run_on_scheduler(
                        lambda dask_scheduler: [str(key) for key in dask_scheduler.tasks]
                    )
                    if batches := [key for key in keys if key.startswith("_work-")]:
                        seen.extend(batches)
                        time.sleep(_HELD_S)
                        return
                    time.sleep(0.2)
            finally:
                release.set()

        releaser = threading.Thread(target=_release_once_the_fit_waits, daemon=True)
        releaser.start()

        async def _fit_and_judge():
            async with M._petab_server(work, _build_root(scenario), args, scenario=scenario) as s:
                call = _make_call(s.session)
                bounds = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}
                fit = await call("make_changes", parameters=[bounds], justification="x", n_starts=2)
                meta = _base_meta(scenario, args)
                judged = await _judge(
                    _Judged(scenario.example_dir),
                    call,
                    _Baseline(),
                    work,
                    meta,
                    _ToolClock(),
                    time.monotonic(),
                )
                return fit, judged

        fit, (_, result) = asyncio.run(_fit_and_judge())
        releaser.join(timeout=60)
        assert held.result(timeout=60) is True

    assert seen, "the fit never reached the pool"
    assert isinstance(fit, dict) and fit.get("committed"), fit
    assert result["fit_settings"]["PETAB_MCP_FIT_POOL"] == address
    assert _HELD_S / 2 <= result["queue_seconds"] < result["fit_seconds"], result
    assert (
        json.loads((work / "result.json").read_text())["queue_seconds"] == result["queue_seconds"]
    )


def test_a_pool_from_another_environment_ends_the_run_as_fit_pool(tmp_path, monkeypatch):
    """#105's third check, at the server's startup: its version check refuses a pool whose
    scheduler imports another amici (PEtab-MCP#226), as after a pin bump made without restarting
    the pool. The server exits with the report in its log alone, before the run has a session, and
    the run stops on it as the pool's failure rather than as an abort."""
    scenario = _scenario(tmp_path)
    monkeypatch.setattr(M, "archive_run", lambda *args: None)
    args = build_parser(scenario).parse_args(["--driver", "scripted"])
    with _pool(tmp_path, amici="0.0.0") as address:
        monkeypatch.setenv("PETAB_MCP_FIT_POOL", address)
        monkeypatch.setenv("PETAB_MCP_BUILD_ROOT", str(tmp_path / "build"))
        code, result = asyncio.run(M._main_async(scenario, args))

    assert code == 1
    assert result["stop_reason"] == I.stop_reason(I.FIT_POOL)
    assert "amici 0.0.0 there" in result["infrastructure_evidence"]
    assert result["fit_settings"]["PETAB_MCP_FIT_POOL"] == address


def test_a_pool_without_its_build_root_is_refused_before_anything_runs(tmp_path, monkeypatch):
    """The pool's workers import every module the run's server builds, by its absolute path, so a
    task-private ``.amici_build`` would end every fit as ``fit_pool``. Refused as
    ``petab-mcp --fit-pool`` refuses it, before the working copy is touched."""
    scenario = _scenario(tmp_path)
    monkeypatch.setenv("PETAB_MCP_FIT_POOL", "tcp://127.0.0.1:1")
    monkeypatch.delenv("PETAB_MCP_BUILD_ROOT", raising=False)
    with pytest.raises(ValueError, match="PETAB_MCP_BUILD_ROOT"):
        run_scenario(scenario, ["--driver", "scripted"])
    assert not (scenario.example_dir / "work").exists()


def test_the_build_root_is_the_pools_or_else_the_examples(tmp_path, monkeypatch):
    """#105's second check, for the build root: without a pool or a root, it is the example's
    ``.amici_build``, as before."""
    scenario = _scenario(tmp_path)
    monkeypatch.delenv("PETAB_MCP_FIT_POOL", raising=False)
    monkeypatch.delenv("PETAB_MCP_BUILD_ROOT", raising=False)
    assert _build_root(scenario) == scenario.example_dir / ".amici_build"
    monkeypatch.setenv("PETAB_MCP_FIT_POOL", "tcp://cn081:8786")
    monkeypatch.setenv("PETAB_MCP_BUILD_ROOT", str(tmp_path / "pool_builds"))
    assert _build_root(scenario) == tmp_path / "pool_builds"
