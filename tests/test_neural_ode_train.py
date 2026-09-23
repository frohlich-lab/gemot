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
    # k is what PEtab estimates besides the weights: the latent's two starts (the readout scale
    # factor is fixed by the staging).
    assert result.n_ode_parameters == 2
    assert set(result.parameters) == {"z1_0_c_lo", "z1_0_c_hi"}
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
    assert result.n_ode_parameters == 0 and result.parameters == {}
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


def test_fides_restarts_are_not_failures_and_share_one_compiled_gradient(tmp_path, monkeypatch):
    """Two defects this pins, both invisible to a single-restart fit and so to every other test.

    A fides EXIT FLAG IS NOT A FAILURE. ``FTOL``/``GTOL``/``XTOL`` are it converging and
    ``MAXITER`` is it spending its budget; reporting any of them down the fourth return value made
    the caller discard the fit as an aborted restart.

    And the jitted value-and-gradient BELONGS TO THE STEPPER, built once for the run. Closing a
    fresh one over each restart's ``unravel`` compiles an XLA executable per restart, and enough of
    them exhaust ``vm.max_map_count`` -- the ``LLVM compilation error: Cannot allocate memory``
    abort ``pysr_worker`` documents, which killed a fifty-restart fit outright.
    """
    import sys

    import equinox as eqx

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    jits = []
    real_filter_jit = eqx.filter_jit

    def counting(*a, **k):
        # The STEPPER's own calls and nothing else. diffrax, optimistix and amici's `JAXModel`
        # apply `filter_jit` at IMPORT time -- nineteen calls between them -- so whether those
        # land inside this window depends only on what an earlier test in the same xdist worker
        # already imported. Counting them made this pass under the full suite, where something
        # imports diffrax first, and fail whenever the module was run on its own: 21 against a
        # budget of 4, from a run whose stepper compiled exactly the 2 it should.
        if sys._getframe(1).f_globals.get("__name__") == T.__name__:
            jits.append(1)
        return real_filter_jit(*a, **k)

    monkeypatch.setattr(T.eqx, "filter_jit", counting)
    restarts = 4
    config = T.TrainConfig(width=4, depth=1, steps=20, n_restarts=restarts, patience=1000,
                           optimiser="fides")
    result = T.train(problem, tmp_path / "run", config)

    assert not result.solver_failed and math.isfinite(result.nll)
    assert result.restart_failures == ()          # a converged restart is not a failed one
    assert result.config.key().endswith("_fides_s0")
    # The compilations belong to the stepper, one per curriculum stage -- NOT one per restart, so
    # the count must not grow with `restarts`.
    assert len(jits) < restarts, f"{len(jits)} jit compilations for {restarts} restarts"


def test_the_horizon_curriculum_grows_the_window_and_carries_the_weights(tmp_path):
    """Stages come from `petab_sciml.training.CurriculumLearningProblem`, so the standard owns the
    partition. What this checks is the two things the library does NOT do for us: its export drops
    the SciML extension's files, without which a stage cannot be imported at all, and it truncates
    an observable out of the early stages entirely."""
    import pandas as pd

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, steps=21, n_restarts=1, horizon_stages=3, patience=1000)
    result = T.train(problem, tmp_path / "run", config)
    assert config.key() == "aug0_w4d1_cl3_s0" and result.notes == ()
    # One objective per stage, and the weights carry across them: the loop seeds `networks` once
    # per RESTART, not per stage, so a stage starts where the previous one ended.
    assert len(result.stage_objectives) == 3 and result.steps_run == 21
    assert result.stage_objectives[-1] == result.nll and math.isfinite(result.nll)

    stages = sorted((tmp_path / "run" / "cl_prob").glob("stage*"), key=lambda p: p.name)
    assert len(stages) == 3
    rows = [len(pd.read_csv(s / "measurements.tsv", sep="\t")) for s in stages]
    assert rows == sorted(rows) and rows[0] < rows[-1], "the horizon must grow"
    full = len(pd.read_csv(tmp_path / "run" / "measurements.tsv", sep="\t"))
    assert rows[-1] == full, "the last stage is the whole time course"
    for stage in stages:
        # The export writes the PEtab tables only; these are copied beside them, and a stage
        # without them cannot be imported -- which is why the copy is not belt-and-braces.
        assert (stage / "net1.py").exists() and (stage / "net1_ps.hdf5").exists()
        assert (stage / "hybridization.tsv").exists()
        # An observable dropped by truncation would take its noise parameters out of use, which
        # lint reports as extraneous. Every stage keeps them all.
        assert pd.read_csv(stage / "measurements.tsv", sep="\t")["observableId"].nunique() == 2


def test_a_curriculum_is_one_kind_or_the_other(tmp_path):
    """Both build a curriculum from one step budget. The version that let `n_windows` silently win
    would run a sweep arm labelled as a horizon curriculum that had never built one."""
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    with pytest.raises(ValueError, match="both build a"):
        T.train(problem, tmp_path / "run",
                T.TrainConfig(width=4, depth=1, steps=9, n_restarts=1, n_windows=2, horizon_stages=3))


def test_the_saved_bundle_carries_the_fitted_weights_not_the_staged_ones(tmp_path):
    """`save_bundle` is what makes a trained field outlive its process, and it had no test at all.
    Staging writes the network's initial weights; nothing writes them again, so a bundle that
    merely copied the staged directory would ship a random field that scores like a trained one."""
    import h5py
    import numpy as np

    from petab_harness.neural_ode import stage as S

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    result = T.train(problem, tmp_path / "run", T.TrainConfig(width=4, depth=1, steps=8, n_restarts=1,
                                                              patience=1000))
    bundle = T.save_bundle(result, tmp_path / "bundle")

    def arrays(path):
        out = {}
        with h5py.File(path) as handle:
            handle.visititems(lambda name, o: out.__setitem__(name, np.array(o))
                              if getattr(o, "shape", None) else None)
        return out

    staged, saved = arrays(result.staged.directory / "net1_ps.hdf5"), arrays(bundle / "net1_ps.hdf5")
    assert staged and set(staged) == set(saved), "the bundle must hold the same layers"
    assert any(not np.allclose(staged[k], saved[k]) for k in staged), \
        "the bundle holds the STAGED weights: the fit was thrown away"
    # And what it holds is the fit itself, layer for layer, not merely something different.
    trained = result.problem.model.nns[S.NETWORK_ID].layers
    for name, layer in trained.items():
        assert np.allclose(saved[f"parameters/{S.NETWORK_ID}/{name}/weight"], np.asarray(layer.weight))
        assert np.allclose(saved[f"parameters/{S.NETWORK_ID}/{name}/bias"], np.asarray(layer.bias))


def test_checkpoints_record_the_best_so_far_and_reconstruct_a_scorable_result(tmp_path):
    """A checkpoint must be what a SHORTER run would have returned, not the current iterate."""
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, n_augmented=1, steps=20, n_restarts=2, patience=1000,
                           checkpoints=(5, 10, 20))
    result = T.train(problem, tmp_path / "run", config)
    assert not result.solver_failed

    # One per (restart, checkpoint), in the order taken.
    assert [(c.restart, c.steps) for c in result.checkpoints] == [
        (0, 5), (0, 10), (0, 20), (1, 5), (1, 10), (1, 20)]
    # Best-so-far can only improve along a restart, and the last must equal what the restart
    # reached -- the whole point is that checkpoint k stands in for a run of length k.
    for restart in (0, 1):
        along = [c.nll for c in result.checkpoints if c.restart == restart]
        assert along == sorted(along, reverse=True)
    assert min(c.nll for c in result.checkpoints) == pytest.approx(result.nll)

    # The reconstruction is scorable: it carries the checkpoint's parameters, not the run's.
    early = next(c for c in result.checkpoints if c.restart == 0 and c.steps == 5)
    stand_in = T.at_checkpoint(result, early)
    assert stand_in.nll == early.nll and stand_in.parameters == early.parameters
    assert stand_in.staged is result.staged and stand_in.config is result.config
    assert stand_in.checkpoints == ()
    assert T.trajectories(stand_in, n_times=4)["c_lo"]["states"].shape == (4, 5)


def test_checkpoints_are_refused_where_a_step_is_not_a_budget(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    with pytest.raises(ValueError, match="checkpoints need optimiser='adam'"):
        T.train(problem, tmp_path / "a",
                T.TrainConfig(width=4, depth=1, steps=20, optimiser="fides", checkpoints=(5,)))
    with pytest.raises(ValueError, match="fall outside"):
        T.train(problem, tmp_path / "b",
                T.TrainConfig(width=4, depth=1, steps=20, checkpoints=(5, 999)))


def test_no_checkpoints_is_the_default_and_costs_nothing(tmp_path):
    """The negative control: without the flag the run is unchanged and records nothing."""
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    plain = T.TrainConfig(width=4, depth=1, n_augmented=1, steps=20, n_restarts=1, patience=1000)
    marked = T.TrainConfig(width=4, depth=1, n_augmented=1, steps=20, n_restarts=1, patience=1000,
                           checkpoints=(5, 10, 20))
    a, b = T.train(problem, tmp_path / "a", plain), T.train(problem, tmp_path / "b", marked)
    assert a.checkpoints == () and len(b.checkpoints) == 3
    # Checkpointing must not perturb the trajectory: same seed, same optimum, same key.
    assert a.nll == pytest.approx(b.nll) and a.steps_run == b.steps_run
    assert a.config.key() == b.config.key()


def test_an_early_stop_truncates_the_curve_rather_than_faking_the_tail(tmp_path):
    """Patience ends the run, so checkpoints past the stop describe no run and must not appear."""
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, n_augmented=1, steps=40, n_restarts=1,
                           patience=1, min_improvement=1e9, checkpoints=(2, 20, 40))
    result = T.train(problem, tmp_path / "run", config)
    taken = result.steps_run
    assert taken < 40, "patience=1 with an unreachable min_improvement should stop almost at once"
    # A checkpoint AT the stopping step still fires -- it precedes the patience check, and a run
    # of exactly that budget would have returned exactly that optimum. Later ones must not.
    assert [c.steps for c in result.checkpoints] == [s for s in (2, 20, 40) if s <= taken]
