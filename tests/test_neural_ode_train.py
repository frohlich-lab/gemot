"""Training through AMICI's JAX backend: a finite fit, the ODE's own k, dense trajectories."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from test_neural_ode_problem import write_problem

from petab_harness.neural_ode import problem as P
from petab_harness.neural_ode import train as T


def test_training_fits_counts_and_simulates(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, n_augmented=1, steps=30, n_restarts=2, patience=1000)
    result = T.train(problem, tmp_path / "run", config)
    assert not result.solver_failed and math.isfinite(result.nll) and result.restart_failures == ()
    # k is what PEtab estimates besides the weights: the scale factor and the latent's two starts.
    assert result.n_ode_parameters == 3
    assert set(result.parameters) == {"scale_Q", "z1_0_c_lo", "z1_0_c_hi"}
    assert all(v > 0 for v in result.parameters.values())
    assert result.n_network_weights == result.staged.n_weights and result.steps_run == 60
    assert result.config.key() == "aug1_w4d1_s0"
    assert result.stage_objectives == (result.nll,) and result.notes == ()

    trajectories = T.trajectories(result, n_times=10)
    assert set(trajectories) == {"c_lo", "c_hi"}
    assert trajectories["c_lo"]["states"].shape == (10, 5) and T.state_names(result) == ("P", "Q", "S", "E", "z1")
    assert trajectories["c_hi"]["times"][-1] == pytest.approx(problem.horizon)
    # The dosed starts are the condition table's, so the simulated start is the dose.
    assert trajectories["c_hi"]["states"][0, 2] == pytest.approx(8.0)


def test_curriculum_multiple_shooting_runs_its_stages_and_ends_single_shooting(tmp_path):
    import pandas as pd

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, steps=21, n_restarts=1, n_windows=2, patience=1000)
    result = T.train(problem, tmp_path / "run", config)
    assert config.key() == "aug0_w4d1_cms2_s0" and result.notes == ()
    # Two stages: a third of the steps on the two-window stage, the rest single shooting.
    assert len(result.stage_objectives) == 2 and result.steps_run == 21
    assert result.stage_objectives[-1] == result.nll and math.isfinite(result.nll)
    # The final problem is the staged one: its k has no window starts, the stage before does.
    assert result.n_ode_parameters == 1 and set(result.parameters) == {"scale_Q"}
    stage1 = tmp_path / "run" / "curriculum" / "stage1"
    parameters = pd.read_csv(stage1 / "parameters.tsv", sep="\t").set_index("parameterId")
    windows = [p for p in parameters.index if p.startswith("WINDOW1_")]
    assert windows and all("_PARAMETER_" in p for p in windows) and (stage1 / "net1.py").exists()
    # Five time points split [0, 0.5] and [0.5, 2]; a window start begins at the datum measured
    # where the window starts (P at t=0.5 in c_hi is 0.9), an unmeasured state at its guess.
    assert pd.read_csv(stage1 / "experiments.tsv", sep="\t").query("experimentId == 'WINDOW1_EXPERIMENT_c_hi'")["time"].min() == 0.5
    assert float(parameters.loc["WINDOW1_EXPERIMENT_c_hi_PARAMETER_P", "nominalValue"]) == pytest.approx(0.9)
    assert float(parameters.loc["WINDOW1_EXPERIMENT_c_hi_PARAMETER_S", "nominalValue"]) == pytest.approx(
        0.1 * problem.scales["S"])


def test_windows_are_ignored_under_preequilibration_and_reduced_on_sparse_data(tmp_path):
    problem = P.load(write_problem(tmp_path / "pre", preequilibration=True), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, steps=5, n_restarts=1, n_windows=3, patience=1000)
    result = T.train(problem, tmp_path / "run", config)
    assert len(result.stage_objectives) == 1 and result.notes and "refuses preequilibration" in result.notes[0]
    # Five time points have four gaps, so six windows cannot all have length; the count drops.
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    stages, notes = T._curriculum(problem, T.stage_only(problem, tmp_path / "sparse", config), replace(config, n_windows=6))
    assert 2 <= len(stages) <= 4 and notes and "n_windows=6 reduced to" in notes[0]
