"""Distillation: samples where the trajectories went, fronts into a graded family, Antimony in data units."""

from __future__ import annotations

import re

import antimony
import numpy as np
import pytest
import sympy as sp
from test_neural_ode_problem import write_problem

from petab_harness.neural_ode import distil as D
from petab_harness.neural_ode import problem as P
from petab_harness.neural_ode import pysr_worker as W
from petab_harness.neural_ode import train as T


def _trained(tmp_path, *, protocol: bool = False):
    problem = P.load(write_problem(tmp_path / "toy", protocol=protocol), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, steps=6, n_restarts=1, patience=1000)
    return problem, T.train(problem, tmp_path / "run", config)


def test_samples_are_drawn_along_the_trajectories_in_the_networks_units(tmp_path):
    problem, result = _trained(tmp_path, protocol=True)
    features, derivatives, variables = D.sample(problem, result, D.DistilConfig(n_samples=50, n_times=20, seed=1))
    assert variables == result.staged.field_inputs == ("P", "Q", "S", "E", "pulse")
    assert features.shape == (50, 5) and derivatives.shape == (50, 4) and np.isfinite(features).all()
    # The protocol column is its formula at each sample's time and experiment: the experiment's
    # drug before t=1 (0 or 5), zero after.
    assert set(np.round(features[:, 4], 6)) <= {0.0, 5.0}
    # States are in the network's units: divided by their scales, so O(1).
    assert np.all(np.abs(features[:, :4]) < 20)


def test_fronts_become_a_graded_family_of_antimony_models(tmp_path, monkeypatch):
    problem, result = _trained(tmp_path)

    def fake_regress(features, derivatives, placeholders, config):
        assert placeholders == ["x0", "x1", "x2", "x3", "x4"]  # P, Q, S, E, drug
        return [{"selected": "0.5*x2 - 0.25*x0", "r2": 0.9, "front": [[3, "0.5*x2", 0.5], [5, "0.5*x2 - 0.25*x0", 0.9]]},
                {"selected": "0.25*x0", "r2": 0.8, "front": [[3, "0.25*x0", 0.8]]},
                {"selected": "-0.5*x2", "r2": 0.95, "front": [[3, "-0.5*x2", 0.95]]},
                {"selected": "x3*x4", "r2": 0.7, "front": [[3, "x3*x4", 0.7]]}]

    monkeypatch.setattr(D, "_regress", fake_regress)
    distilled = D.distil(problem, result, D.DistilConfig(n_samples=20, n_times=10))
    assert distilled.pysr_available and distilled.state_names == ("P", "Q", "S", "E")
    assert [str(e) for e in distilled.expressions] == ["-0.25*P + 0.5*S", "0.25*P", "-0.5*S", "E*drug"]
    assert distilled.field_r2 == (0.9, 0.8, 0.95, 0.7)
    model = distilled.antimony
    # Species keep their names, so the shipped observables apply; inputs are parameters the
    # condition table sets; every fitted number is a named constant, integers stay structure.
    assert model.startswith("model toy_distilled\n  species P, Q, S, E;") and "  drug = 0;" in model
    rules = [line for line in model.splitlines() if "' = " in line]
    assert len(rules) == 4 and not any(re.search(r"\d\.\d", rule) for rule in rules)
    assert distilled.n_constants == sum(1 for line in model.splitlines() if re.match(r"  c\d+ = ", line))
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(model) >= 0, antimony.getLastError()
    # Two budgets: at 3 every state takes its cheapest row; at 5 only P can spend more.
    assert [c.complexity_budget for c in distilled.candidates] == [3, 5]
    assert distilled.candidates[1].expressions[0] == distilled.expressions[0]
    assert distilled.candidates[0].complexities == (3, 3, 3, 3)


def test_constants_are_named_once_per_magnitude_and_integers_are_structure():
    expression, constants = D._name_constants(sp.sympify("2*x - 0.5*y + 0.5*z + 0.125*w"), [])
    assert dict(constants) == {"c0": 0.5, "c1": 0.125}
    w, x, y, z, c0, c1 = sp.symbols("w x y z c0 c1")
    # One name per magnitude, the sign carried; the stoichiometric 2 stays a 2.
    assert expression.coeff(y) == -c0 and expression.coeff(z) == c0 and expression.coeff(w) == c1
    assert expression.coeff(x) == 2


def test_without_pysr_the_result_says_so_and_the_worker_refuses(tmp_path, monkeypatch):
    problem, result = _trained(tmp_path)

    def unavailable(*args, **kwargs):
        raise D.RegressionUnavailable("no Julia here")

    monkeypatch.setattr(D, "_regress", unavailable)
    distilled = D.distil(problem, result, D.DistilConfig(n_samples=10, n_times=5))
    assert not distilled.pysr_available and distilled.antimony == "" and "no Julia here" in distilled.warnings[0]
    assert distilled.candidates == () and distilled.n_constants == 0
    if W.pysr_class() is None:
        with pytest.raises(ImportError, match="PySR is not installed"):
            W.regress(np.zeros((3, 2)), np.zeros((3, 1)), ["x0", "x1"], {"workers": 1})


def test_a_readout_state_is_emitted_as_a_species_under_its_own_id_and_an_assignment_rule(tmp_path):
    """A ``readout_*`` name belongs to a measurement mapping, so the state cannot carry it: the
    server refuses a problem that declares one as a species, which rejected every proposal on
    every blinded benchmark example."""
    problem = P.load(write_problem(tmp_path / "toy", protocol=False), dosed=("S",))
    names = ("readout_P", "Q")

    class staged:  # noqa: N801 -- a stand-in for the trained problem's units
        field_inputs = (*names, *problem.field_input_names)
        scales = dict.fromkeys(field_inputs, 1.0)
        horizon = 1.0

    expressions = (-sp.Float(0.5) * sp.Symbol("readout_P") + sp.Symbol("Q"), sp.Float(0.25) * sp.Symbol("readout_P"))
    model, _constants = D.emit(problem, staged, names, expressions, {"readout_P": 2.0, "Q": 1.0})
    assert "  species s_P, Q;" in model and "readout_P" not in model.split("// measurement")[0]
    assert "  readout_P := s_P;" in model
    # The rate rules are on the emitted species, and mention the readout name nowhere.
    rules = [line for line in model.splitlines() if "' = " in line]
    assert [line.split("'")[0].strip() for line in rules] == ["s_P", "Q"]
    assert all("readout_" not in rule for rule in rules) and all("s_P" in rule for rule in rules)
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(model) >= 0, antimony.getLastError()
