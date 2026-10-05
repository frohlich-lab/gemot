"""The server-free tier's fakes answer exactly as the real server does.

Every test outside ``tests/server/`` reads the server through ``tests/fakes/server_api.py``. A
fake that drifted from the module it stands in for would keep those tests green while the code
they cover misjudged real runs. So each fake is run here against the real module on the same
inputs, edge cases first: a pin bump that changes a criterion, a sigma predicate or the scaling
resolver fails here, and the fix is to the fake (or to the gemot code the change breaks).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import petab.v1 as petab
import pytest

from conftest import YAML_PATH
from fakes import server_api as fake

from gemot import server_api

_NUMBERS = [None, float("nan"), float("inf"), "x", -1, 0, 1, 2, 3, 5, 40]
_NLLS = [None, float("nan"), -3.5, 0.0, 12.25, 1e6]


def _same(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


# ----------------------------------------------------------------------------- constants
def test_the_fake_constants_are_the_servers():
    """Every constant the fake stands in for has the server's value: the judge's noise-sigma gate
    reads these, and must agree with the server's adequacy gate on what an implausible sigma is."""
    fake_constants = fake.constants()
    names = [n for n in vars(fake_constants) if n.isupper()]
    assert names, "the fake defines no constants"
    real = server_api.constants()
    for name in names:
        assert getattr(fake_constants, name) == getattr(real, name), name


# ------------------------------------------------------------------------------- fit settings
_FIT_SETTINGS = ("FIT_PER_START_TIMEOUT_S", "FIT_N_WORKERS", "FIT_AMICI_ATOL")


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"PETAB_MCP_FIT_N_WORKERS": "3", "PETAB_MCP_FIT_PER_START_TIMEOUT": "60"},
        {"PETAB_MCP_FIT_N_WORKERS": "0", "PETAB_MCP_FIT_PER_START_TIMEOUT": "nan"},
        {"PETAB_MCP_FIT_N_WORKERS": "x", "PETAB_MCP_FIT_PER_START_TIMEOUT": "-5"},
        {"SLURM_CPUS_PER_TASK": "4"},
        {"SLURM_CPUS_PER_TASK": "4", "PETAB_MCP_FIT_N_WORKERS": "2"},
        {"PETAB_MCP_FIT_AMICI_ATOL": "1e-8"},
        {"PETAB_MCP_FIT_AMICI_ATOL": "inf"},
    ],
)
def test_the_fake_fit_settings_resolve_as_the_servers_do(env):
    """Both resolve from the environment ONCE, at import, so each environment runs in a fresh
    interpreter: what is compared is the rule (overrides, malformed values, the SLURM grant), not
    one machine's numbers."""
    tests = Path(__file__).resolve().parents[1]
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "from fakes import server_api as fake\n"
        "from petab_mcp import fitting\n"
        "for name in %r:\n"
        "    print(name, getattr(fake.fitting(), name), getattr(fitting, name))\n"
    ) % (str(tests), _FIT_SETTINGS)
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PETAB_MCP_FIT_") and k != "SLURM_CPUS_PER_TASK"
    }
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env={**clean, **env}
    )
    assert out.returncode == 0, out.stderr
    for line in out.stdout.splitlines():
        name, mine, theirs = line.split()
        assert mine == theirs, f"{env}: {name} fake={mine} server={theirs}"


# ------------------------------------------------------------------------------ criteria
def test_the_fake_criteria_are_the_servers():
    real = server_api.criteria()
    assert set(fake.PENALTIES) == set(real.PENALTIES)
    assert set(fake.CANONICAL) == set(real.CANONICAL)
    for name in real.PENALTIES:
        mine, theirs = fake.CANONICAL[name], real.CANONICAL[name]
        assert mine.name == theirs.name
        for nll in _NLLS:
            for k in (None, -1, 0, 1, 3, 7):
                for n in (None, 0, 1, 2, 3, 8, 40):
                    for k_eff in (None, float("nan"), 2.5):
                        assert _same(
                            mine.value(nll, k, n, k_eff), theirs.value(nll, k, n, k_eff)
                        ), (
                            name,
                            nll,
                            k,
                            n,
                            k_eff,
                        )
                        assert _same(mine.penalty(k, n, k_eff), theirs.penalty(k, n, k_eff))


@pytest.mark.parametrize("name", [None, "", "  ", "bic", " AICc ", "tic", "nonsense", 3])
def test_the_fake_criterion_reads_a_reported_name_as_the_server_does(name):
    mine = fake.Criterion.from_name(name)
    theirs = server_api.criteria().Criterion.from_name(name)
    assert (mine is None) == (theirs is None)
    if mine is not None:
        assert mine.name == theirs.name


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"bic": 10.0},
        {"bic": float("nan"), "criterion": "bic", "criterion_value": 9.0},
        {"criterion": "aicc", "criterion_value": 4.0},
        {"aicc": 3.0, "criterion": "aicc", "criterion_value": 4.0},
        {"criterion": "bic", "criterion_value": "x"},
    ],
)
def test_the_fake_criterion_reads_a_stored_score_as_the_server_does(record):
    for name in fake.PENALTIES:
        mine = fake.Criterion(name).score_of(record)
        theirs = server_api.criteria().Criterion(name).score_of(record)
        assert _same(mine, theirs), (name, record)


@pytest.mark.parametrize("env", [None, "aic", "nonsense"])
@pytest.mark.parametrize("arg", [None, "caic"])
def test_the_fake_resolves_a_criterion_as_the_server_does(monkeypatch, env, arg):
    if env is None:
        monkeypatch.delenv("PETAB_MCP_CRITERION", raising=False)
    else:
        monkeypatch.setenv("PETAB_MCP_CRITERION", env)
    assert fake.resolve(arg).name == server_api.criteria().resolve(arg).name


# ------------------------------------------------------------------------------- fitting
_REPORTS = [
    None,
    "not a report",
    {},
    {"s": "not an entry"},
    {"s": {"effective_sigma": 2.0, "upper_bound_cap": 1.0}},
    {"s": {"effective_sigma": 0.5, "upper_bound_cap": 1.0}},
    {"s": {"effective_sigma": None, "fitted": 9.0, "upper_bound_cap": 1.0}},
    {"s": {"fitted": 9.0, "upper_bound_cap": 1.0}},
    {"s": {"fitted": "x", "upper_bound_cap": 1.0}},
    {"s": {"fitted": 9.0, "upper_bound_cap": None}},
    {"s": {"fraction_of_data_std": 0.001}},
    {"s": {"fraction_of_data_std": 0.01}},
    {"s": {"fraction_of_data_std": 0.5}},
    {"s": {"fraction_of_data_std": "0.001"}},
    {"a": {"fraction_of_data_std": 0.5}, "b": {"effective_sigma": 3, "upper_bound_cap": 1}},
]


@pytest.mark.parametrize("report", _REPORTS)
def test_the_fake_sigma_predicates_are_the_servers(report):
    real = server_api.fitting()
    assert fake.noise_sigma_inflated(report) == real.noise_sigma_inflated(report)
    assert fake.noise_sigma_collapsed(report) == real.noise_sigma_collapsed(report)


# ----------------------------------------------------------------------------- reconcile
def test_the_fake_resolves_observable_scalings_as_the_server_does():
    toy = petab.Problem.from_yaml(str(YAML_PATH))
    crafted = petab.Problem(
        observable_df=pd.DataFrame(
            {
                "observableFormula": ["scale_A * A + offset_A", "1e-3 * B", float("nan"), 2.5],
                "noiseFormula": ["sd_A", "sd_B", "1", "1"],
            },
            index=pd.Index(["obs_A", "obs_B", "obs_C", "obs_D"], name="observableId"),
        ),
        measurement_df=pd.DataFrame(
            {
                "observableId": ["obs_A", "obs_A", "obs_B"],
                "observableParameters": ["s1;o1", float("nan"), "0.5"],
            }
        ),
    )
    for problem in (toy, crafted):
        assert fake.observable_scaling_ids(problem) == (
            server_api.reconcile().observable_scaling_ids(problem)
        )
