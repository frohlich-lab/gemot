"""The trained field against a held-out set, in both modes, and the command line that writes the record."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from test_neural_ode_problem import (
    write_problem,
    write_v2_conditions,
    write_v2_experiments,
    write_v2_problem,
)

from gemot.neural_ode import heldout as H
from gemot.neural_ode import problem as P
from gemot.neural_ode import stage as S
from gemot.neural_ode import train as T
from gemot.neural_ode.__main__ import main

_HELD_OUT = """observableId\tsimulationConditionId\ttime\tmeasurement\tnoiseParameters
obs_P\t{cond}\t0.0\t0.05\t
obs_P\t{cond}\t1.0\t0.9\t
obs_P\t{cond}\t2.0\t1.6\t
obs_Q\t{cond}\t1.5\t4.0\t0.2
"""


def write_held_out(
    directory: Path, *, new_condition: bool, transfer: bool = False, extra_column: bool = False
) -> Path:
    """A held-out table beside the fixture: a new condition ``c_new`` that adds a row of the
    training table's inputs, or the training conditions again; ``transfer`` brings a parameter
    table, ``extra_column`` a condition column the training table does not have."""
    directory.mkdir(parents=True, exist_ok=True)
    cond = "c_new" if new_condition else "c_lo"
    (directory / "measurements.tsv").write_text(_HELD_OUT.format(cond=cond))
    if new_condition:
        header = "conditionId\tS\tinit_E\tdrug" + ("\tlight" if extra_column else "")
        (directory / "conditions.tsv").write_text(
            header + "\nc_new\t4.0\t1.0\t2.0" + ("\t1" if extra_column else "") + "\n"
        )
    if transfer:
        (directory / "parameters.tsv").write_text(
            "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
            "sigma_P\tlin\t0.0001\t100.0\t0.1\t0\nscale_Q\tlog10\t0.001\t1000.0\t1.0\t1\nQ_init\tlin\t0.0\t10.0\t0.5\t0\n"
        )
    return directory / "measurements.tsv"


def write_v2_held_out(
    directory: Path,
    *,
    basal: str | None = None,
    dose: str = "init_E",
    switch: bool = False,
    ramp: bool = False,
) -> Path:
    """A native v2 fixed-θ set: experiment ``e_new`` applying ``c_new``, which sets new values of
    the training targets, after ``basal``'s pre-equilibration where one is given; with ``switch``,
    a later period at t=1 sets ``drug`` again, as a protocol moved into the experiment table does;
    with ``ramp``, ``c_new`` also sets ``drug_rate`` 1.5 and ``drug_end`` 1.75, so the entity map's
    protocol (``write_v2_problem(protocol=True)``) ramps from ``drug`` until t=1.75, as fujita's
    held-out ramps do."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "measurements.tsv").write_text(
        "experimentId\tobservableId\ttime\tmeasurement\tnoiseParameters\n"
        "e_new\tobs_P\t0.0\t0.05\t\ne_new\tobs_P\t1.0\t0.9\t\ne_new\tobs_P\t2.0\t1.6\t\n"
        "e_new\tobs_Q\t1.5\t4.0\t0.2\n"
    )
    write_v2_conditions(directory / "conditions.tsv", {"c_new": (4.0, 1.0, 2.0)}, dose)
    write_v2_experiments(directory / "experiments.tsv", {"e_new": "c_new"}, basal)
    if ramp:
        with (directory / "conditions.tsv").open("a") as table:
            table.write("c_new\tdrug_rate\t1.5\nc_new\tdrug_end\t1.75\n")
    if switch:
        with (directory / "conditions.tsv").open("a") as table:
            table.write("c_drug\tdrug\t6.0\n")
        with (directory / "experiments.tsv").open("a") as table:
            table.write("e_new\t1.0\tc_drug\n")
    return directory / "measurements.tsv"


def _trained(tmp_path):
    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    config = T.TrainConfig(width=4, depth=1, steps=8, n_restarts=1, patience=1000)
    return problem, T.train(problem, tmp_path / "run", config)


def test_fixed_theta_scores_a_new_condition_in_the_training_units(tmp_path):
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True))
    payload = H.score(result, held_out, tmp_path / "score", training=problem)
    # P's start is its t=0 datum, Q's the pinned Q_init, S and E the new row's: nothing to refit.
    assert payload["status"] == "ok" and payload["mode"] == "fixed" and payload["n"] == 4
    assert math.isfinite(payload["nll"]) and "k" not in payload
    # Staged in the training design: same rate-rule constants, the trained weights carried over.
    model = (tmp_path / "score" / "staged" / "model.ant").read_text()
    assert f"P' = ({result.staged.scales['P']!r} / 2.0) * f_P;" in model


def test_a_transfer_set_is_refused_rather_than_scored(tmp_path):
    """A transfer set asks the model to re-estimate its parameters on the held-out system. A
    field's parameters are its weights -- 90 to 1400 against 40-odd points -- so the question has
    no answer at this ratio, and freezing them to refit the starts instead answers a different one
    from the reference bar, which DOES re-estimate its own dynamics (crauste's refits all twelve
    rate constants per unit). No number is better than a number that is not comparable."""
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(
        write_held_out(tmp_path / "validation", new_condition=True, transfer=True)
    )
    payload = H.score(result, held_out, tmp_path / "score", training=problem)
    assert payload["status"] == H.STATUS_NO_TRANSFER and payload["mode"] == "transfer"
    assert "nll" not in payload and "bic" not in payload
    assert "weights" in payload["reason"]


def test_a_new_input_column_is_a_different_field(tmp_path):
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(
        write_held_out(tmp_path / "validation", new_condition=True, extra_column=True)
    )
    payload = H.score(result, held_out, tmp_path / "score", training=problem)
    assert payload["status"] == "table_mismatch" and "light" in payload["reason"]


def test_a_v1_set_is_keyed_by_experiment_as_the_server_keys_it(tmp_path):
    """A v1 set names a simulation condition; the server moves it onto the training problem's
    experiments (`as_v2`) before it scores it, and the field must be scored on that same view, or
    the two scorers would be reading different held-out problems."""
    from petab_mcp.validation_data import as_v2, read_held_out

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True))
    view = H._held_out_problem(problem, held_out)
    (experiment,) = view.experiment_ids
    assert set(as_v2(held_out, problem.tables).measurement_df["experimentId"]) == {experiment}
    assert view.periods[experiment] == ((0.0, ("c_new",)),)
    assert view.condition_values[experiment] == {"S": 4.0, "init_E": 1.0, "drug": 2.0}


def test_an_experiment_neither_table_defines_is_a_table_mismatch(tmp_path):
    from petab_mcp.validation_data import HeldOut

    problem = P.load(write_problem(tmp_path / "toy"), dosed=("S",))
    measurements = pd.DataFrame(
        {"observableId": ["obs_P"], "experimentId": ["e1"], "time": [1.0], "measurement": [0.9]}
    )
    with pytest.raises(P.ProblemError, match="e1"):
        H._held_out_problem(problem, HeldOut(measurement_df=measurements))


def test_the_command_line_writes_the_record_and_the_trained_bundle(tmp_path):
    example = tmp_path / "toy"
    write_problem(example / "problem")
    write_held_out(example / "validation", new_condition=True)
    (example / "problem" / "reference.json").write_text(
        json.dumps(
            {
                "nll": -1.0,
                "k": 2,
                "n": 12,
                "validation": {"status": "ok", "mode": "fixed", "nll": 3.0, "n": 4},
            }
        )
    )
    code = main(
        [
            "fit",
            str(example / "problem"),
            "--out",
            str(tmp_path / "out"),
            "--dosed",
            "S",
            "--width",
            "4",
            "--depth",
            "1",
            "--steps",
            "6",
            "--restarts",
            "1",
        ]
    )
    assert code == 0
    record = json.loads((tmp_path / "out" / "result.json").read_text())
    # k is 0: the toy pins every start (S and E by condition columns, Q by a fixed Q_init, P by
    # its t=0 datum) and `scale_Q` is a readout scale factor, which staging now FIXES at its
    # nominal -- a neural state's magnitude is free, so a scale factor times that state is all the
    # data constrain. Nothing but the weights is left to estimate, and the weights are never k.
    assert record["method"] == "neural_ode" and record["k"] == 0 and record["n"] == 12
    assert record["bic"] == pytest.approx(2 * record["nll"])
    assert record["validation"]["status"] == "ok" and record["validation"]["mode"] == "fixed"
    assert record["validation"]["delta_nll"] == pytest.approx(record["validation"]["nll"] - 3.0)
    assert record["reference"] == {"nll": -1.0, "k": 2, "n": 12}
    # The bundle is the staged problem with the fitted scalars and weights written back. Nothing
    # was estimated here, so the check that bites is that the fixed scale factor came through at
    # its nominal rather than drifting: the readout is pinned, the state carries the magnitude.
    assert "scale_Q" not in record["parameters"]
    parameters = pd.read_csv(tmp_path / "out" / "model" / "parameters.tsv", sep="\t").set_index(
        "parameterId"
    )
    assert float(parameters.loc["scale_Q", "nominalValue"]) == pytest.approx(2.0)
    assert not bool(parameters.loc["scale_Q", "estimate"])
    with h5py.File(tmp_path / "out" / "model" / "net1_ps.hdf5") as weights:
        assert weights["parameters/net1/layer1/weight"].shape == (4, 5)


def test_a_native_v2_problem_and_set_run_end_to_end(tmp_path):
    """The route on PEtab v2 as the benchmark ships it: training tables keyed by experiment, and a
    held-out set that brings its own experiment and long condition table beside its measurements.
    Before the port the loader refused the problem and the scorer refused the set."""
    example = tmp_path / "toy"
    write_v2_problem(example / "problem")
    write_v2_held_out(example / "validation")
    (example / "problem" / "reference.json").write_text(
        json.dumps({"validation": {"status": "ok", "mode": "fixed", "nll": 3.0, "n": 4}})
    )
    out = tmp_path / "out"
    argv = ["fit", str(example / "problem"), "--out", str(out), "--dosed", "S"]
    argv += ["--width", "4", "--depth", "1", "--steps", "6", "--restarts", "1"]
    assert main(argv) == 0
    record = json.loads((out / "result.json").read_text())
    assert record["held_out_path"] == str(example / "validation" / "measurements.tsv")
    assert record["validation"]["status"] == "ok" and record["validation"]["mode"] == "fixed"
    assert record["validation"]["n"] == 4 and math.isfinite(record["validation"]["nll"])
    experiments = pd.read_csv(out / "heldout" / "staged" / "experiments.tsv", sep="\t")
    assert experiments["experimentId"].tolist() == ["e_new"]


def test_a_held_out_ramp_is_the_entity_maps_protocol_at_the_sets_values(tmp_path):
    """fujita's held-out pulses and ramps are its training stimulus at new condition values: the
    protocol is the training problem's, read from the example's entity map, and each held-out
    experiment sets its `EGF_0`, `EGF_rate` and `EGF_end`. The view evaluates it as the distiller
    samples it: from 2 at 1.5 per unit, restarted from 6 by the switch at t=1, then 0 after
    t=1.75."""
    from petab_mcp.validation_data import read_held_out

    from gemot.neural_ode import distil as D

    problem = P.load(write_v2_problem(tmp_path / "toy" / "problem", protocol=True), dosed=("S",))
    held_out = read_held_out(write_v2_held_out(tmp_path / "validation", switch=True, ramp=True))
    view = H._held_out_problem(problem, held_out)
    assert view.timed_inputs == problem.timed_inputs and view.field_input_names == ("stim",)
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    stim = D._protocol_values(view, "stim", times, np.array(["e_new"] * len(times)))
    assert list(stim) == [2.0, 2.75, 7.5, 8.25, 0.0]


def _settling_field(staged: S.Staged, driver: str = "drug") -> list[np.ndarray]:
    """Weights for a width-``len(states)`` field with one stable steady state that the input
    ``driver`` moves: ``f_k = b_k - tanh(x_k / s_k + 0.3 driver / s_driver [k = 0])``. A field drawn
    at random has no steady state to pre-equilibrate to, so a held-out score under
    pre-equilibration needs one built."""
    n = len(staged.states)
    first = np.zeros((n, len(staged.field_inputs)))
    first[:, :n] = np.eye(n)
    first[0, staged.field_inputs.index(driver)] = 0.3
    return [first, np.zeros(n), -np.eye(n), np.array([0.2, -0.1, 0.3, 0.1])]


def _twin(staged: S.Staged, weights, starts: dict[str, str]) -> str:
    """The staged model with the network written out as rate laws: the same ODE as a plain SBML
    model the server can score, with ``starts`` as its states' initial values."""
    first, bias, second, offset = weights
    text = (staged.directory / "model.ant").read_text()
    hidden = [
        "tanh("
        + " + ".join(
            f"{float(w)!r} * {name}_norm" for w, name in zip(row, staged.field_inputs) if w
        )
        + f" + {float(b)!r})"
        for row, b in zip(first, bias)
    ]
    for state, row, b in zip(staged.states, second, offset):
        rate = (
            " + ".join(f"{float(w)!r} * {h}" for w, h in zip(row, hidden) if w) + f" + {float(b)!r}"
        )
        text = text.replace(f"  f_{state} = 0;", f"  f_{state} := {rate};")
    for state, start in starts.items():
        text = text.replace(f"  {state} = 0;", f"  {state} = {start};")
    return text


def test_the_held_out_nll_is_the_servers_at_the_same_point(tmp_path):
    """The field's held-out nll is the one the server's `evaluate_validation_data` computes for the
    same model at the same point -- here the field written out as an SBML twin -- on a native v2
    set that pre-equilibrates, switches its input mid-run as a protocol moved into the experiment
    table does, and drives the field through the entity map's protocol, which ramps and then cuts
    off at condition values only the held-out experiment sets (fujita's EGF). The two read one
    held-out view, built by the server's own functions (`as_v2`, `_with_held_out_design`), and
    differ only in the integrator: AMICI's JAX backend against SUNDIALS, so the tolerance is the
    server's own for solver error. The field reads the protocol's `stim`, so a ramp, a cut-off or
    a switch either side ignored would show in the nll."""
    from types import SimpleNamespace

    import equinox as eqx
    import jax.numpy as jnp
    from petab.v2.models.sbml_model import SbmlModel
    from petab_mcp.problem import WorkingProblem
    from petab_mcp.validation_data import read_held_out, score_problem

    yaml_path = write_v2_problem(
        tmp_path / "toy" / "problem", preequilibration=True, dose="E", protocol=True
    )
    problem = P.load(yaml_path, dosed=("S", "E"))
    config = T.TrainConfig(width=4, depth=1)
    staged = T.stage_only(problem, tmp_path / "field", config)
    assert staged.field_inputs[-1] == "stim"
    held_out = read_held_out(
        write_v2_held_out(
            tmp_path / "validation", basal="c_basal", dose="E", switch=True, ramp=True
        )
    )
    prepared = H.prepare(
        SimpleNamespace(staged=staged, config=config),
        held_out,
        tmp_path / "score",
        training=problem,
    )
    assert not isinstance(prepared, dict), prepared
    # The starts the route stages: P's pre-equilibration start is the one scalar it estimates.
    assert prepared.ids == ["P_0_c_basal"]
    periods = pd.read_csv(prepared.staged.directory / "experiments.tsv", sep="\t")
    assert periods["conditionId"].tolist() == ["start_c_basal", "c_new", "c_drug"]

    weights = _settling_field(staged, "stim")
    networks = prepared.imported.model.nns
    net = eqx.tree_at(
        lambda n: (
            n.layers["layer1"].weight,
            n.layers["layer1"].bias,
            n.layers["layer2"].weight,
            n.layers["layer2"].bias,
        ),
        networks[S.NETWORK_ID],
        tuple(jnp.asarray(w) for w in weights),
    )
    route = H.score_prepared(prepared, {**networks, S.NETWORK_ID: net}, {"P_0_c_basal": 0.3})

    twin = WorkingProblem.from_yaml(yaml_path)
    model = tmp_path / "twin.ant"
    model.write_text(_twin(staged, weights, {"P": "0.3", "Q": "Q_init;\n  Q_init = 0.5"}))
    twin.model = SbmlModel.from_antimony(model)
    server = score_problem(
        twin,
        held_out.measurement_df,
        condition_df=held_out.condition_df,
        experiment_df=held_out.experiment_df,
        build_dir=tmp_path / "amici",
        model_name="twin",
    )
    assert route["status"] == server["status"] == "ok", (route, server)
    assert route["n"] == server["n"] == 4 and server["n_experiments"] == 1
    assert route["nll"] == pytest.approx(server["nll"], rel=1e-4)


def test_a_curve_of_checkpoints_imports_the_held_out_set_once(tmp_path, monkeypatch):
    """A budget curve scores every checkpoint against one held-out set. Preparing once must stage
    and import once, however many fields are then scored against it."""
    from gemot.neural_ode import train as T
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True))

    imported = []
    real = T._import
    monkeypatch.setattr(T, "_import", lambda *a, **k: (imported.append(1), real(*a, **k))[1])

    prepared = H.prepare(result, held_out, tmp_path / "curve", training=problem)
    assert not isinstance(prepared, dict), prepared
    payloads = [
        H.score_prepared(prepared, result.problem.model.nns, result.parameters) for _ in range(6)
    ]

    assert len(imported) == 1, f"one import for six scorings, got {len(imported)}"
    assert all(p["status"] == "ok" and p["mode"] == "fixed" for p in payloads)
    # Same field six times must give the same score: the context carries no state between calls.
    assert len({round(p["nll"], 9) for p in payloads}) == 1
    # And the convenience wrapper still agrees with the prepared path.
    assert H.score(result, held_out, tmp_path / "once", training=problem)["nll"] == pytest.approx(
        payloads[0]["nll"]
    )


def test_a_prepared_set_refuses_a_transfer_as_score_does(tmp_path):
    """Preparing is not a way round the fixed-theta rule: a transfer set is refused up front."""
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(
        write_held_out(tmp_path / "validation", new_condition=True, transfer=True)
    )
    prepared = H.prepare(result, held_out, tmp_path / "curve", training=problem)
    assert isinstance(prepared, dict) and prepared["mode"] == "transfer"


def test_scoring_different_fields_against_one_prepared_set_does_not_reuse_a_score(tmp_path):
    """The reuse must not go too far: a different field has to give a different number, or the
    curve would be flat by construction and would 'prove' whatever it was pointed at."""
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True))
    prepared = H.prepare(result, held_out, tmp_path / "curve", training=problem)

    import jax.tree_util as jtu

    perturbed = jtu.tree_map(
        lambda x: x * 1.5 if hasattr(x, "shape") else x, result.problem.model.nns
    )
    a = H.score_prepared(prepared, result.problem.model.nns, result.parameters)
    b = H.score_prepared(prepared, perturbed, result.parameters)
    assert a["status"] == "ok" and b["status"] == "ok"
    assert a["nll"] != pytest.approx(b["nll"]), "a changed field must move the held-out score"
