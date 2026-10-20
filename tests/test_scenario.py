"""What a Scenario hands the server it launches: its AMICI tolerances, in the server's environment,
and what the run records of that environment."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

from gemot import fit_budget
from gemot.runner import main as M
from gemot.runner import provenance as P
from gemot.scenario import AMICI_TOLERANCE_ENV, Scenario


async def _noop(call, n):
    return None


class _Spawned(Exception):
    """Raised by the fake Popen once it has seen the launch, so no server is started."""


def _launch_env(tmp_path: Path, scenario: Scenario, monkeypatch) -> dict[str, str]:
    """The environment ``_petab_server`` launches the server with, for ``scenario``."""
    seen = {}

    def popen(command, *, stdout, stderr, env):
        seen.update(env)
        raise _Spawned

    monkeypatch.setattr(M.subprocess, "Popen", popen)
    work = tmp_path / "work"
    work.mkdir()
    args = Namespace(reference_validation="off", n_starts=1)

    async def launch():
        async with M._petab_server(work, tmp_path / "build", args, scenario=scenario):
            pass

    with pytest.raises(_Spawned):
        asyncio.run(launch())
    return seen


def test_a_problems_amici_tolerances_reach_the_servers_environment(tmp_path, monkeypatch):
    """raimundez's likelihood needs a tighter steady-state tolerance than the server's default,
    and its reference bar is fitted under it. Every fit and held-out score the candidate gets runs
    in the launched server, so the tolerance has to be in that process's environment, or the two
    sides of the Δ are integrated differently."""
    scenario = Scenario(
        name="raimundez",
        problem_dir=tmp_path / "problem",
        scripted_fix=_noop,
        amici_tolerances={"relative_tolerance_steady_state": 1e-9, "absolute_tolerance": 1e-12},
    )
    env = _launch_env(tmp_path, scenario, monkeypatch)
    assert env["PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE"] == "1e-09"
    assert env["PETAB_MCP_FIT_AMICI_ATOL"] == "1e-12"
    assert "PETAB_MCP_FIT_AMICI_RTOL" not in env
    assert "PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE" not in env


def test_no_tolerances_export_nothing(tmp_path, monkeypatch):
    """The default leaves the server on its own defaults: an exported variable would pin a value
    the server's default is meant to own."""
    for name in AMICI_TOLERANCE_ENV.values():
        monkeypatch.delenv(name, raising=False)
    scenario = Scenario(name="toy", problem_dir=tmp_path / "problem", scripted_fix=_noop)
    assert scenario.amici_tolerances == {} and scenario.amici_tolerance_env() == {}
    env = _launch_env(tmp_path, scenario, monkeypatch)
    assert not set(AMICI_TOLERANCE_ENV.values()) & set(env)


def test_a_misspelt_tolerance_is_refused_where_it_is_written(tmp_path):
    """A key the server does not read would export nothing and leave the default in place, silently."""
    with pytest.raises(ValueError, match="relative_tolerance_steadystate"):
        Scenario(
            name="raimundez",
            problem_dir=tmp_path / "problem",
            scripted_fix=_noop,
            amici_tolerances={"relative_tolerance_steadystate": 1e-9},
        )


def test_the_run_records_the_fit_settings_its_server_is_launched_with(tmp_path, monkeypatch):
    """result.json recorded `n_starts` and no other fit setting, so sweeps of one example at two
    per-start caps pooled in every aggregate (gemot#6). The record is resolved from the very
    environment the server is launched with, as the server resolves it: the problem's tolerance
    over an inherited one, an inherited one kept, an unusable override at the server's default,
    and the criterion, which the judge read only off a fit."""
    monkeypatch.setenv("PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE", "1e-6")
    monkeypatch.setenv("PETAB_MCP_FIT_AMICI_RTOL", "1e-7")
    monkeypatch.delenv("PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE", raising=False)
    monkeypatch.setenv("PETAB_MCP_FIT_TASK_SECONDS", "-1")
    monkeypatch.setenv("PETAB_MCP_CRITERION", "AICc")
    scenario = Scenario(
        name="raimundez",
        problem_dir=tmp_path / "problem",
        scripted_fix=_noop,
        amici_tolerances={"relative_tolerance_steady_state": 1e-9},
    )
    args = Namespace(driver="scripted", model=None, recursion_limit=1, n_starts=1, temperature=0.0)
    recorded = P._base_meta(scenario, args)["fit_settings"]

    launched = _launch_env(tmp_path, scenario, monkeypatch)
    assert recorded == {**fit_budget.server_fit_settings(launched), "PETAB_MCP_CRITERION": "aicc"}
    assert recorded["PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE"] == 1e-9
    assert recorded["PETAB_MCP_FIT_AMICI_RTOL"] == 1e-7
    assert recorded["PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE"] is None
    assert recorded["PETAB_MCP_FIT_TASK_SECONDS"] == 270.0
    assert recorded["PETAB_MCP_FIT_PER_START_TIMEOUT"] == fit_budget.FIT_PER_START_TIMEOUT_S
    assert recorded["PETAB_MCP_FIT_N_WORKERS"] == fit_budget.FIT_N_WORKERS
