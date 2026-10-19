"""What the neural route reads off a PEtab problem's tables, either version, and what it refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemot.neural_ode import problem as P

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
obs_Q\tc_lo\t1.0\t3.0\t{sigma_q}{preeq_lo}
obs_Q\tc_hi\t2.0\t6.0\t{sigma_q}{preeq_hi}
"""
_OBSERVABLES = """observableId\tobservableFormula\tnoiseFormula\tobservableTransformation
obs_P\tP\tsigma_P\tlin
obs_Q\tscale_Q * Q\tnoiseParameter1_obs_Q\t{transformation}
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
#: The sigmas as an estimation problem states them, nominals far from any start (perelson's 1e5).
_ESTIMATED_SIGMAS = """parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate
sigma_P\tlog10\t0.0001\t1000000.0\t100000.0\t1
sigma_Q\tlog10\t0.0001\t1000000.0\t100000.0\t1
scale_Q\tlog10\t0.001\t1000.0\t2.0\t1
Q_init\tlin\t0.0\t10.0\t0.5\t0
"""
_INPUTS = """inputId\tformula\tconditionInputs
pulse\tpiecewise(drug, time < 1, 0)\tdrug
"""
_YAML = """format_version: {version}
parameter_file: parameters.tsv
problems:
- sbml_files: [model.xml]
  condition_files: [conditions.tsv]
  measurement_files: [measurements.tsv]
  observable_files: [observables.tsv]
"""
_EMPTY_SBML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" '
    'level="3" version="2"><model id="empty"/></sbml>\n'
)


def write_problem(
    directory: Path,
    *,
    preequilibration: bool = False,
    protocol: bool = False,
    partial_preequilibration: bool = False,
    version: int = 1,
    log10: bool = False,
    estimate_sigma: bool = False,
) -> Path:
    """A zero-shot v1 problem: two observed states (one behind a scale factor), a dosed substrate
    ``S`` the convention cannot see, a dosed ``init_E``, a clamped ``drug`` and an empty model."""
    directory.mkdir(parents=True, exist_ok=True)
    preeq = preequilibration or partial_preequilibration
    (directory / "measurements.tsv").write_text(
        _MEASUREMENTS.format(
            preeq="\tpreequilibrationConditionId" if preeq else "",
            preeq_lo="\tc_basal" if preeq else "",
            preeq_hi="\t" if partial_preequilibration else ("\tc_basal" if preeq else ""),
            sigma_q="sigma_Q" if estimate_sigma else "0.2",
        )
    )
    (directory / "observables.tsv").write_text(
        _OBSERVABLES.format(transformation="log10" if log10 else "lin")
    )
    (directory / "conditions.tsv").write_text(
        _CONDITIONS.format(extra="c_basal\t0.0\t1.0\t0.0\n" if preeq else "")
    )
    (directory / "parameters.tsv").write_text(_ESTIMATED_SIGMAS if estimate_sigma else _PARAMETERS)
    (directory / "model.xml").write_text(_EMPTY_SBML)
    if protocol:
        (directory / "inputs.tsv").write_text(_INPUTS)
    (directory / "problem.yaml").write_text(_YAML.format(version=version))
    return directory / "problem.yaml"


_V2_OBSERVABLES = """observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\tobservablePlaceholders\tnoisePlaceholders
obs_P\tP\tsigma_P\tnormal\t\t
obs_Q\tscale_Q * Q\tsd_Q\tnormal\t\tsd_Q
"""
_V2_PARAMETERS = """parameterId\tlowerBound\tupperBound\tnominalValue\testimate
sigma_P\t0.0001\t100.0\t0.1\tfalse
scale_Q\t0.001\t1000.0\t2.0\ttrue
Q_init\t0.0\t10.0\t0.5\tfalse
"""
#: fujita's EGF ramp over the toy's ``drug``: a protocol over condition targets alone.
_RAMP = """inputId\tformula\tconditionInputs
stim\tpiecewise(drug_rate*time + drug, time <= drug_end, 0)\tdrug_rate;drug;drug_end
"""
_V2_YAML = """format_version: 2.0.0
parameter_files: [parameters.tsv]
model_files:
  model: {location: model.xml, language: sbml}
measurement_files: [measurements.tsv]
condition_files: [conditions.tsv]
experiment_files: [experiments.tsv]
observable_files: [observables.tsv]
"""


def write_v2_conditions(path: Path, rows: dict[str, tuple[float, float, float]], dose: str) -> None:
    """A long v2 condition table: per condition, ``S``, the ``dose`` target and ``drug``."""
    path.write_text(
        "conditionId\ttargetId\ttargetValue\n"
        + "".join(
            f"{c}\t{t}\t{v}\n"
            for c, values in rows.items()
            for t, v in zip(("S", dose, "drug"), values)
        )
    )


def write_v2_experiments(path: Path, experiments: dict[str, str], basal: str | None) -> None:
    path.write_text(
        "experimentId\ttime\tconditionId\n"
        + "".join(
            (f"{e}\t-inf\t{basal}\n" if basal else "") + f"{e}\t0.0\t{c}\n"
            for e, c in experiments.items()
        )
    )


def write_v2_problem(
    directory: Path, *, preequilibration: bool = False, dose: str = "init_E", protocol: bool = False
) -> Path:
    """The v1 toy written natively in PEtab v2: experiments ``e_lo``/``e_hi`` applying ``c_lo``/
    ``c_hi``, after ``c_basal`` with ``preequilibration``. obs_Q's noise placeholder is ``sd_Q``,
    declared rather than named by v1's convention. ``dose`` is the target that doses ``E``:
    ``init_E`` as in the v1 toy, or the state ``E`` itself. With ``protocol``, every condition also
    sets ``drug_rate`` 0 and ``drug_end`` 10, as fujita's step doses set ``EGF_rate`` and
    ``EGF_end``, and ``inputs.tsv`` ramps ``stim`` over the three, as fujita's EGF."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = [line.split("\t") for line in _MEASUREMENTS.splitlines()[1:]]
    (directory / "measurements.tsv").write_text(
        "experimentId\tobservableId\ttime\tmeasurement\tnoiseParameters\n"
        + "".join(
            f"e_{r[1].removeprefix('c_')}\t{r[0]}\t{r[2]}\t{r[3]}\t{'0.2' if r[0] == 'obs_Q' else ''}\n"
            for r in rows
        )
    )
    (directory / "observables.tsv").write_text(_V2_OBSERVABLES)
    conditions = {"c_lo": (2.0, 1.0, 0.0), "c_hi": (8.0, 1.0, 5.0)}
    if preequilibration:
        conditions["c_basal"] = (0.0, 1.0, 0.0)
    write_v2_conditions(directory / "conditions.tsv", conditions, dose)
    if protocol:
        with (directory / "conditions.tsv").open("a") as table:
            table.write("".join(f"{c}\tdrug_rate\t0.0\n{c}\tdrug_end\t10.0\n" for c in conditions))
        (directory / "inputs.tsv").write_text(_RAMP)
    write_v2_experiments(
        directory / "experiments.tsv",
        {"e_lo": "c_lo", "e_hi": "c_hi"},
        "c_basal" if preequilibration else None,
    )
    (directory / "parameters.tsv").write_text(_V2_PARAMETERS)
    (directory / "model.xml").write_text(_EMPTY_SBML)
    (directory / "problem.yaml").write_text(_V2_YAML)
    return directory / "problem.yaml"


def test_states_knobs_starts_and_scales_are_read_off_the_tables(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    assert problem.observed == ("P", "Q") and problem.dosed == ("S", "E")
    assert problem.state_names == ("P", "Q", "S", "E") and problem.inputs == ("drug",)
    assert problem.start_columns == {"S": "S", "init_E": "E"}
    # The v1 problem is upgraded on load: one experiment per simulation condition, in the order
    # the measurements name them.
    lo, hi = problem.experiment_ids
    assert problem.periods == {lo: ((0.0, ("c_lo",)),), hi: ((0.0, ("c_hi",)),)}
    assert not problem.uses_preequilibration and problem.switches == {}
    # A fixed Q_init pins Q; P's start is its t=0 datum, one per experiment.
    assert problem.pinned_starts == {"Q": 0.5}
    assert problem.measured_starts == {(lo, "P"): 0.0, (hi, "P"): 0.1}
    # sigma_P is fixed and obs_Q's noise is numeric, so the problem estimates no sigma; scale_Q
    # is the one readout scale factor the table estimates.
    assert problem.noise_parameters == () and problem.observable_parameters == ("scale_Q",)
    assert problem.n_measurements == 12
    # Q is scaled by scale_Q (nominal 2), so its magnitude is that of the data divided by it.
    assert problem.scales["Q"] == pytest.approx(((1.5**2 + 3.0**2) / 2) ** 0.5)
    assert problem.scales["S"] == pytest.approx((34.0) ** 0.5) and problem.scales["E"] == 1.0
    assert problem.horizon == 2.0 and problem.warnings == ()
    counts = problem.counts(shared_ics=False)
    # Known starts: P (measured, both experiments), Q (pinned), S and E (columns) -- 8 of 8.
    assert counts["n_fixed_ics"] == 8 and problem.counts(shared_ics=True)["n_fixed_ics"] == 4
    assert counts["n_modelled_states"] == 2 and counts["n_dosed"] == 2 and counts["n_inputs"] == 1


def test_a_protocol_replaces_the_columns_it_reads_and_hidden_structure_is_refused(tmp_path):
    """The server's own check refuses a protocol over a fitted constant, as it refuses the
    problem."""
    problem = P.load(write_problem(tmp_path / "toy", protocol=True), dosed=("S",))
    assert [t.id for t in problem.timed_inputs] == ["pulse"]
    # drug is consumed by the protocol; the field reads the protocol's value instead.
    assert problem.field_input_names == ("pulse",) and problem.scales["pulse"] == 1.0
    leaky = write_problem(tmp_path / "leak", protocol=True)
    (leaky.parent / "inputs.tsv").write_text(_INPUTS + "leak\tk_hidden * time\tk_hidden\n")
    with pytest.raises(P.ProblemError, match="'leak' names k_hidden"):
        P.load(leaky, dosed=("S",))


def test_preequilibration_binds_every_experiment_or_none(tmp_path):
    problem = P.load(write_problem(tmp_path / "pre", preequilibration=True), dosed=("S",))
    assert problem.preequilibration == dict.fromkeys(problem.experiment_ids, "c_basal")
    assert problem.basal_conditions == ("c_basal",) and problem.uses_preequilibration
    with pytest.raises(P.ProblemError, match="preequilibrate in no condition"):
        P.load(write_problem(tmp_path / "partial", partial_preequilibration=True), dosed=("S",))


def test_a_v1_problem_is_upgraded_on_load_and_reads_as_its_native_v2_twin(tmp_path):
    """There is one upgrade, the server's (`WorkingProblem.from_yaml`), and the route reads what it
    gives. A v1 problem and the same problem written natively in v2 are the same problem to the
    route, whatever the upgrade calls the experiments: a reader that kept its own v1 path would be
    a second conversion that could drift from the server's, which is the likelihood scored."""
    for preequilibration in (False, True):
        root = tmp_path / str(preequilibration)
        v1 = P.load(write_problem(root / "v1", preequilibration=preequilibration), dosed=("S",))
        v2 = P.load(write_v2_problem(root / "v2", preequilibration=preequilibration), dosed=("S",))
        assert v2.experiment_ids == ("e_lo", "e_hi")
        same = dict(zip(v1.experiment_ids, v2.experiment_ids))
        for name in ("observed", "dosed", "inputs", "start_columns", "pinned_starts", "scales"):
            assert getattr(v1, name) == getattr(v2, name), name
        for name in ("noise_parameters", "observable_parameters", "horizon", "n_measurements"):
            assert getattr(v1, name) == getattr(v2, name), name
        assert {same[e]: v for e, v in v1.preequilibration.items()} == v2.preequilibration
        assert {(same[e], s): v for (e, s), v in v1.measured_starts.items()} == v2.measured_starts
        assert {same.get(k, k): v for k, v in v1.condition_values.items()} == v2.condition_values
        assert {same[e]: v for e, v in v1.periods.items()} == v2.periods
    assert v2.preequilibration == {"e_lo": "c_basal", "e_hi": "c_basal"}
    assert v2.periods["e_hi"] == ((float("-inf"), ("c_basal",)), (0.0, ("c_hi",)))


def test_a_later_period_that_sets_an_input_again_is_followed(tmp_path):
    """A v2 experiment may switch an input mid-run (a protocol `assignments_to_experiments` moved
    into the experiment table). The field reads it through AMICI's events; the route's own
    bookkeeping -- the input's scale and the distiller's sample -- must follow it too, or the
    distilled rate law is regressed against the dose before the switch.

    ``e_hi`` has the shape the server's conversion gives a protocol that starts later: an empty
    period at t=0, so the experiment starts where a v1 simulation did, then the switch. Its knobs
    read as zero until then, and that is no reason for a warning."""
    yaml_path = write_v2_problem(tmp_path / "toy")
    with (yaml_path.parent / "conditions.tsv").open("a") as table:
        table.write("c_drug\tdrug\t9.0\n")
    (yaml_path.parent / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\n"
        "e_lo\t0.0\tc_lo\ne_lo\t1.0\tc_drug\ne_hi\t0.0\t\ne_hi\t1.0\tc_hi\n"
    )
    problem = P.load(yaml_path, dosed=("S",))
    assert problem.switches == {
        "e_lo": ((1.0, {"S": 2.0, "init_E": 1.0, "drug": 9.0}),),
        "e_hi": ((1.0, {"S": 8.0, "init_E": 1.0, "drug": 5.0}),),
    }
    assert list(problem.input_values("e_lo", "drug", [0.0, 0.5, 1.0, 2.0])) == [0, 0, 9, 9]
    assert list(problem.input_values("e_hi", "drug", [0.0, 2.0])) == [0.0, 5.0]
    assert problem.periods["e_hi"] == ((0.0, ()), (1.0, ("c_hi",)))
    assert problem.warnings == () and ("e_hi", "S") not in problem.column_starts
    assert problem.scales["drug"] == pytest.approx(((0.0**2 * 2 + 5.0**2 + 9.0**2) / 4) ** 0.5)


def test_an_offset_does_not_enter_a_states_scale():
    from petab.v1.math import sympify_petab

    assert P._gain(sympify_petab("offset + 2 * Q"), "Q", {"offset": 5.0}, None) == 2.0
    assert (
        P._gain(
            sympify_petab("observableParameter1_o * Q"), "Q", {}, "0.5", ["observableParameter1_o"]
        )
        == 0.5
    )


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
    lo, hi = problem.experiment_ids
    assert problem.start_columns.get("init_E") == "E"
    assert problem.start_is_known(lo, "E")
    assert not problem.start_is_known(hi, "E"), "a blank cell pins nothing"
    # And the count follows: E is fixed in one experiment, not both.
    assert (lo, "E") in problem.column_starts and (hi, "E") not in problem.column_starts
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
        observables.replace("obs_P\tP\tsigma_P", "obs_P\tP\tsigma_P * exp(-decay_P * time)")
    )
    parameters = (yaml_path.parent / "parameters.tsv").read_text()
    (yaml_path.parent / "parameters.tsv").write_text(
        parameters + "decay_P\tlin\t0.0\t10.0\t0.1\t1\n"
    )

    problem = P.load(yaml_path, dosed=("S",))
    assert "exp" not in problem.noise_parameters, problem.noise_parameters
    assert "time" not in problem.noise_parameters
    # The real estimated symbols in the formula still count.
    assert "decay_P" in problem.noise_parameters
    assert problem.counts(shared_ics=False)["n_sigma"] == len(problem.noise_parameters)
