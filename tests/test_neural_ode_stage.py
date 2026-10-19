"""The staged PEtab SciML problem: lint-clean, wired end to end, every start accounted for."""

from __future__ import annotations

import math
import statistics

import h5py
import pandas as pd
import pytest
import sympy as sp
import yaml
from test_neural_ode_problem import write_problem

from gemot.neural_ode import problem as P
from gemot.neural_ode import stage as S


def _conditions(staged: S.Staged) -> dict[tuple[str, str], str]:
    conditions = pd.read_csv(staged.directory / "conditions.tsv", sep="\t")
    return {(r.conditionId, r.targetId): str(r.targetValue) for r in conditions.itertuples()}


def _starts(staged: S.Staged) -> dict[tuple[str, str], str]:
    """``(experiment, target) -> value`` of the condition each experiment starts in."""
    experiments = pd.read_csv(staged.directory / "experiments.tsv", sep="\t")
    first = experiments.groupby("experimentId", sort=False)["conditionId"].first()
    rows = _conditions(staged)
    return {(e, t): v for e, c in first.items() for (cid, t), v in rows.items() if cid == c}


def test_the_staged_problem_is_lint_clean_and_wired_end_to_end(tmp_path):
    from petab import v2

    problem = P.load(write_problem(tmp_path / "toy", protocol=True), dosed=("S",))
    staged = S.stage(problem, tmp_path / "staged", width=4, depth=2, n_augmented=1, clock=True)
    # petab's own reading of the result, extension included, finds nothing wrong with it.
    issues = v2.Problem.from_yaml(str(staged.yaml)).validate()
    assert [i.message for i in issues if i.level.name == "ERROR"] == []

    assert staged.states == ("P", "Q", "S", "E", "z1")
    assert staged.field_inputs == ("P", "Q", "S", "E", "z1", "clock", "pulse")
    hybrid = pd.read_csv(staged.directory / "hybridization.tsv", sep="\t")
    assert list(hybrid["targetValue"][:7]) == [f"{s}_norm" for s in staged.field_inputs]
    assert list(hybrid["targetId"][7:]) == [f"f_{s}" for s in staged.states]
    model = (staged.directory / "model.ant").read_text()
    assert "pulse := piecewise(drug, time < 1, 0);" in model and "clock' = 1;" in model
    assert f"Q' = ({problem.scales['Q']!r} / 2.0) * f_Q;" in model
    with h5py.File(staged.directory / "net1_ps.hdf5") as weights:
        assert weights["parameters/net1/layer1/weight"].shape == (4, 7)
        assert weights["parameters/net1/layer3/weight"].shape == (5, 4)
    assert staged.n_weights == (7 * 4 + 4) + (4 * 4 + 4) + (4 * 5 + 5)

    # Starts: the dosing targets renamed to the states they dose, P's t=0 data, the pinned Q_init
    # as a number, the latent as one estimated parameter per experiment.
    lo, hi = problem.experiment_ids
    starts = _starts(staged)
    assert starts[(lo, "S")] == "2.0" and starts[(hi, "E")] == "1.0"
    assert starts[(lo, "P")] == "0.0" and starts[(hi, "P")] == "0.1"
    assert starts[(lo, "Q")] == "0.5" and starts[(hi, "z1")] == f"z1_0_{hi}"
    assert staged.estimated_starts == (f"z1_0_{lo}", f"z1_0_{hi}")
    parameters = pd.read_csv(staged.directory / "parameters.tsv", sep="\t").set_index("parameterId")
    assert set(parameters.index) == {"sigma_P", "scale_Q", "net1_ps", f"z1_0_{lo}", f"z1_0_{hi}"}
    assert parameters.loc["net1_ps", "nominalValue"] == "array"
    assert (
        parameters.loc[f"z1_0_{lo}", "upperBound"] == 10.0
        and str(parameters.loc["sigma_P", "estimate"]).lower() == "false"
    )
    # The readout scale factor is fixed at its nominal: the state carries the magnitude.
    assert (
        str(parameters.loc["scale_Q", "estimate"]).lower() == "false"
        and float(parameters.loc["scale_Q", "nominalValue"]) == 2.0
    )
    document = yaml.safe_load(staged.yaml.read_text())
    assert document["extensions"]["sciml"]["neural_networks"]["net1"]["format"] == "equinox"
    # Experiments and measurements keep the ids the server's upgrade gave them; each experiment
    # starts in one condition that carries both the shipped knobs and the starts.
    experiments = pd.read_csv(staged.directory / "experiments.tsv", sep="\t")
    assert experiments["experimentId"].tolist() == [lo, hi]
    assert experiments["conditionId"].tolist() == [f"start_{lo}", f"start_{hi}"]
    assert set(pd.read_csv(staged.directory / "measurements.tsv", sep="\t")["experimentId"]) == {
        lo,
        hi,
    }


def test_preequilibration_starts_sit_on_the_basal_condition_and_are_always_staged(tmp_path):
    problem = P.load(write_problem(tmp_path / "pre", preequilibration=True), dosed=("S",))
    native = S.stage(problem, tmp_path / "native", width=4)
    experiments = pd.read_csv(native.directory / "experiments.tsv", sep="\t")
    assert experiments["conditionId"].tolist() == ["start_c_basal", "c_lo", "start_c_basal", "c_hi"]
    assert [math.isinf(t) for t in experiments["time"].astype(float)] == [True, False, True, False]
    # The settled state carries over, so no start sits on c_lo or c_hi; the basal start condition
    # has the basal knobs, S, E and Q from the tables, and P estimated once for the one basal key.
    conditions = _conditions(native)
    assert not any(c in ("c_lo", "c_hi") and s in ("P", "Q") for c, s in conditions)
    assert conditions[("start_c_basal", "P")] == "P_0_c_basal"
    assert conditions[("start_c_basal", "Q")] == "0.5"
    assert conditions[("start_c_basal", "drug")] == "0.0"
    assert native.estimated_starts == ("P_0_c_basal",)

    # There is no way to ask for the periods to be dropped: a problem that preequilibrates says
    # its experiments begin at a steady state, and staging without that fits a different estimand
    # whose likelihood cannot be compared with the reference's.
    with pytest.raises(TypeError, match="preequilibrate"):
        S.stage(problem, tmp_path / "dropped", width=4, preequilibrate=False)
    again = S.stage(problem, tmp_path / "again", width=4)
    periods = pd.read_csv(again.directory / "experiments.tsv", sep="\t")
    assert [math.isinf(t) for t in periods["time"].astype(float)] == [True, False, True, False]


def test_log10_noise_keeps_the_v1_likelihood_under_v2s_log_normal(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy", log10=True), dosed=("S",))
    staged = S.stage(problem, tmp_path / "staged", width=4)
    observables = pd.read_csv(staged.directory / "observables.tsv", sep="\t").set_index(
        "observableId"
    )
    # The server's upgrade writes log-normal; the noise formula carries ln 10, ONCE, so the density
    # is v1's.
    assert observables.loc["obs_Q", "noiseDistribution"] == "log-normal"
    noise = sp.sympify(observables.loc["obs_Q", "noiseFormula"])
    assert sp.simplify(noise / sp.Symbol("noiseParameter1_obs_Q") - sp.log(10)) == 0
    assert (
        observables.loc["obs_P", "noiseDistribution"] == "normal"
        and observables.loc["obs_P", "noiseFormula"] == "sigma_P"
    )


def test_an_estimated_sigma_starts_at_its_datas_spread(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy", log10=True, estimate_sigma=True), dosed=("S",))
    staged = S.stage(problem, tmp_path / "staged", width=4)
    parameters = pd.read_csv(staged.directory / "parameters.tsv", sep="\t").set_index("parameterId")
    # sigma_P scales every obs_P datum through the formula; sigma_Q reaches obs_Q through the
    # measurement override, and obs_Q is log10, so its spread is in decades. Neither is the table's 1e5.
    assert float(parameters.loc["sigma_P", "nominalValue"]) == pytest.approx(
        statistics.pstdev([0.0, 0.3, 0.5, 0.7, 0.8, 0.1, 0.9, 1.5, 2.0, 2.4])
    )
    assert float(parameters.loc["sigma_Q", "nominalValue"]) == pytest.approx(
        statistics.pstdev([math.log10(3.0), math.log10(6.0)])
    )
    assert str(parameters.loc["sigma_Q", "estimate"]).lower() == "true"


def test_shared_starts_are_one_parameter_per_state(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    staged = S.stage(
        problem, tmp_path / "shared", width=4, n_augmented=2, shared_initial_conditions=True
    )
    assert staged.estimated_starts == ("z1_0", "z2_0")
    lo, hi = problem.experiment_ids
    assert _starts(staged)[(lo, "z2")] == _starts(staged)[(hi, "z2")] == "z2_0"
