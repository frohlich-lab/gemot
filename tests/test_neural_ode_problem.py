"""What the neural route reads off a PEtab v1 problem's tables, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from petab_harness.neural_ode import problem as P

_MEASUREMENTS = """observableId\tsimulationConditionId\ttime\tmeasurement\tnoiseParameters{preeq}
obs_P\tc_lo\t0.0\t0.0\t{preeq_lo}
obs_P\tc_lo\t0.5\t0.3\t{preeq_lo}
obs_P\tc_lo\t1.0\t0.5\t{preeq_lo}
obs_P\tc_lo\t1.5\t0.7\t{preeq_lo}
obs_P\tc_lo\t2.0\t0.8\t{preeq_lo}
obs_P\tc_hi\t0.0\t0.1\t{preeq_hi}
obs_P\tc_hi\t0.5\t0.9\t{preeq_hi}
obs_P\tc_hi\t1.0\t1.5\t{preeq_hi}
obs_P\tc_hi\t1.5\t2.0\t{preeq_hi}
obs_P\tc_hi\t2.0\t2.4\t{preeq_hi}
obs_Q\tc_lo\t1.0\t3.0\t0.2{preeq_lo}
obs_Q\tc_hi\t2.0\t6.0\t0.2{preeq_hi}
"""
_OBSERVABLES = """observableId\tobservableFormula\tnoiseFormula
obs_P\tP\tsigma_P
obs_Q\tscale_Q * Q\tnoiseParameter1_obs_Q
"""
_CONDITIONS = """conditionId\tS\tinit_E\tdrug
c_lo\t2.0\t1.0\t0.0
c_hi\t8.0\t1.0\t5.0
{extra}"""
_PARAMETERS = """parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate
sigma_P\tlin\t0.0001\t100.0\t0.1\t0
scale_Q\tlog10\t0.001\t1000.0\t2.0\t1
Q_init\tlin\t0.0\t10.0\t0.5\t0
"""
_INPUTS = """inputId\tformula\tconditionInputs
pulse\tpiecewise(drug, time < 1, 0)\tdrug
leak\tk_hidden * time\t
"""
_YAML = """format_version: {version}
parameter_file: parameters.tsv
problems:
- sbml_files: [model.xml]
  condition_files: [conditions.tsv]
  measurement_files: [measurements.tsv]
  observable_files: [observables.tsv]
"""
_EMPTY_SBML = ('<?xml version="1.0" encoding="UTF-8"?>\n<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
               'level="3" version="2"><model id="empty"/></sbml>\n')


def write_problem(directory: Path, *, preequilibration: bool = False, protocol: bool = False,
                  partial_preequilibration: bool = False, version: int = 1) -> Path:
    """A zero-shot v1 problem: two observed states (one behind a scale factor), a dosed substrate
    ``S`` the convention cannot see, a dosed ``init_E``, a clamped ``drug`` and an empty model."""
    directory.mkdir(parents=True, exist_ok=True)
    preeq = preequilibration or partial_preequilibration
    (directory / "measurements.tsv").write_text(_MEASUREMENTS.format(
        preeq="\tpreequilibrationConditionId" if preeq else "",
        preeq_lo="\tc_basal" if preeq else "", preeq_hi="\t" if partial_preequilibration else ("\tc_basal" if preeq else "")))
    (directory / "observables.tsv").write_text(_OBSERVABLES)
    (directory / "conditions.tsv").write_text(_CONDITIONS.format(extra="c_basal\t0.0\t1.0\t0.0\n" if preeq else ""))
    (directory / "parameters.tsv").write_text(_PARAMETERS)
    (directory / "model.xml").write_text(_EMPTY_SBML)
    if protocol:
        (directory / "inputs.tsv").write_text(_INPUTS)
    (directory / "problem.yaml").write_text(_YAML.format(version=version))
    return directory / "problem.yaml"


def test_states_knobs_starts_and_scales_are_read_off_the_tables(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    assert problem.observed == ("P", "Q") and problem.dosed == ("S", "E")
    assert problem.state_names == ("P", "Q", "S", "E") and problem.inputs == ("drug",)
    assert problem.start_columns == {"S": "S", "init_E": "E"}
    # A fixed Q_init pins Q; P's start is its t=0 datum, one per experiment.
    assert problem.pinned_starts == {"Q": 0.5}
    assert problem.measured_starts == {("c_lo", "P"): 0.0, ("c_hi", "P"): 0.1}
    assert problem.experiment_ids == ("c_lo", "c_hi") and not problem.uses_preequilibration
    # sigma_P is fixed and obs_Q's noise is numeric, so the problem estimates no sigma.
    assert problem.noise_parameters == () and problem.n_measurements == 12
    # Q is scaled by scale_Q (nominal 2), so its magnitude is that of the data divided by it.
    assert problem.scales["Q"] == pytest.approx(((1.5**2 + 3.0**2) / 2) ** 0.5)
    assert problem.scales["S"] == pytest.approx((34.0) ** 0.5) and problem.scales["E"] == 1.0
    assert problem.horizon == 2.0 and problem.warnings == ()
    counts = problem.counts(shared_ics=False)
    # Known starts: P (measured, both experiments), Q (pinned), S and E (columns) -- 8 of 8.
    assert counts["n_fixed_ics"] == 8 and problem.counts(shared_ics=True)["n_fixed_ics"] == 4
    assert counts["n_modelled_states"] == 2 and counts["n_dosed"] == 2 and counts["n_inputs"] == 1


def test_a_protocol_replaces_the_columns_it_reads_and_hidden_structure_is_refused(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy", protocol=True), dosed=("S",))
    assert [t.target for t in problem.timed_inputs] == ["pulse"]
    # drug is consumed by the protocol; the field reads the protocol's value instead.
    assert problem.field_input_names == ("pulse",) and problem.scales["pulse"] == 1.0
    assert any("leak" in w and "k_hidden" in w for w in problem.warnings)


def test_preequilibration_binds_every_experiment_or_none(tmp_path):
    problem = P.load(write_problem(tmp_path / "pre", preequilibration=True), dosed=("S",))
    assert problem.preequilibration == {"c_lo": "c_basal", "c_hi": "c_basal"}
    assert problem.basal_conditions == ("c_basal",) and problem.uses_preequilibration
    with pytest.raises(P.ProblemError, match="preequilibrate in no condition"):
        P.load(write_problem(tmp_path / "partial", partial_preequilibration=True), dosed=("S",))


def test_a_v2_problem_is_refused(tmp_path):
    with pytest.raises(P.ProblemError, match="not a PEtab v1 problem"):
        P.load(write_problem(tmp_path / "v2", version=2))


def test_a_start_column_with_a_hole_pins_only_the_experiments_it_fills(tmp_path):
    """A start column pins a start where its CELL carries a value, not wherever the column exists.

    `_condition_values` reads a blank cell as zero -- the same as any missing knob -- and zero is
    a legitimate start, so the column's presence cannot stand in for the cell being filled. Left
    unchecked, the blank experiment counts as a fixed IC, which understates `k` and lets the
    latent ceiling grant states the data cannot pay for.
    """
    yaml_path = write_problem(tmp_path / "toy")
    table = yaml_path.parent / "conditions.tsv"
    # Written as text, the way an incomplete sheet really arrives: c_lo keeps its start, c_hi's
    # cell is empty. (pandas will not put "" into a float column, which is the point.)
    table.write_text(table.read_text().replace("c_hi\t8.0\t1.0\t5.0", "c_hi\t8.0\t\t5.0"))
    assert "c_hi\t8.0\t\t5.0" in table.read_text()

    problem = P.load(yaml_path, dosed=("S",))
    assert problem.start_columns.get("init_E") == "E"
    assert problem.start_is_known("c_lo", "E")
    assert not problem.start_is_known("c_hi", "E"), "a blank cell pins nothing"
    # And the count follows: E is fixed in one experiment, not both.
    assert ("c_lo", "E") in problem.column_starts and ("c_hi", "E") not in problem.column_starts
    # Under shared starts a state counts only if EVERY experiment pins it, so E must not count.
    per_experiment = problem.counts(shared_ics=False)["n_fixed_ics"]
    shared = problem.counts(shared_ics=True)["n_fixed_ics"]
    full = P.load(write_problem(tmp_path / "full"), dosed=("S",))
    assert per_experiment < full.counts(shared_ics=False)["n_fixed_ics"]
    assert shared < full.counts(shared_ics=True)["n_fixed_ics"]


def test_a_function_in_a_noise_formula_is_not_a_noise_parameter(tmp_path):
    """`sigma * exp(...)` must not make `exp` a sigma: no parameter row carries it, so the
    estimated-by-default rule would count it and name a parameter that does not exist."""
    yaml_path = write_problem(tmp_path / "toy")
    observables = (yaml_path.parent / "observables.tsv").read_text()
    (yaml_path.parent / "observables.tsv").write_text(
        observables.replace("obs_P\tP\tsigma_P", "obs_P\tP\tsigma_P * exp(-decay_P * time)"))
    parameters = (yaml_path.parent / "parameters.tsv").read_text()
    (yaml_path.parent / "parameters.tsv").write_text(
        parameters + "decay_P\tlin\t0.0\t10.0\t0.1\t1\n")

    problem = P.load(yaml_path, dosed=("S",))
    assert "exp" not in problem.noise_parameters, problem.noise_parameters
    assert "time" not in problem.noise_parameters
    # The real estimated symbols in the formula still count.
    assert "decay_P" in problem.noise_parameters
    assert problem.counts(shared_ics=False)["n_sigma"] == len(problem.noise_parameters)
