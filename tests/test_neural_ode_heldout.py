"""The trained field against a held-out set, in both modes, and the command line that writes the record."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import pandas as pd
import pytest
from test_neural_ode_problem import write_problem

from petab_harness.neural_ode import heldout as H
from petab_harness.neural_ode import problem as P
from petab_harness.neural_ode import train as T
from petab_harness.neural_ode.__main__ import main

_HELD_OUT = """observableId\tsimulationConditionId\ttime\tmeasurement\tnoiseParameters
obs_P\t{cond}\t0.0\t0.05\t
obs_P\t{cond}\t1.0\t0.9\t
obs_P\t{cond}\t2.0\t1.6\t
obs_Q\t{cond}\t1.5\t4.0\t0.2
"""


def write_held_out(directory: Path, *, new_condition: bool, transfer: bool = False, extra_column: bool = False) -> Path:
    """A held-out table beside the fixture: a new condition ``c_new`` that adds a row of the
    training table's inputs, or the training conditions again; ``transfer`` brings a parameter
    table, ``extra_column`` a condition column the training table does not have."""
    directory.mkdir(parents=True, exist_ok=True)
    cond = "c_new" if new_condition else "c_lo"
    (directory / "measurements.tsv").write_text(_HELD_OUT.format(cond=cond))
    if new_condition:
        header = "conditionId\tS\tinit_E\tdrug" + ("\tlight" if extra_column else "")
        (directory / "conditions.tsv").write_text(header + "\nc_new\t4.0\t1.0\t2.0" + ("\t1" if extra_column else "") + "\n")
    if transfer:
        (directory / "parameters.tsv").write_text(
            "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
            "sigma_P\tlin\t0.0001\t100.0\t0.1\t0\nscale_Q\tlog10\t0.001\t1000.0\t1.0\t1\nQ_init\tlin\t0.0\t10.0\t0.5\t0\n")
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
    model = (tmp_path / "score" / "staged" / "v1" / "model.ant").read_text()
    assert f"P' = ({result.staged.scales['P']!r} / 2.0) * f_P;" in model


def test_a_transfer_set_is_refused_rather_than_scored(tmp_path):
    """A transfer set asks the model to re-estimate its parameters on the held-out system. A
    field's parameters are its weights -- 90 to 1400 against 40-odd points -- so the question has
    no answer at this ratio, and freezing them to refit the starts instead answers a different one
    from the reference bar, which DOES re-estimate its own dynamics (crauste's refits all twelve
    rate constants per unit). No number is better than a number that is not comparable."""
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True, transfer=True))
    payload = H.score(result, held_out, tmp_path / "score", training=problem)
    assert payload["status"] == H.STATUS_NO_TRANSFER and payload["mode"] == "transfer"
    assert "nll" not in payload and "bic" not in payload
    assert "weights" in payload["reason"]

def test_a_new_input_column_is_a_different_field(tmp_path):
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True, extra_column=True))
    payload = H.score(result, held_out, tmp_path / "score", training=problem)
    assert payload["status"] == "table_mismatch" and "light" in payload["reason"]


def test_the_command_line_writes_the_record_and_the_trained_bundle(tmp_path):
    example = tmp_path / "toy"
    write_problem(example / "problem")
    write_held_out(example / "validation", new_condition=True)
    (example / "problem" / "reference.json").write_text(json.dumps(
        {"nll": -1.0, "k": 2, "n": 12, "validation": {"status": "ok", "mode": "fixed", "nll": 3.0, "n": 4}}))
    code = main(["fit", str(example / "problem"), "--out", str(tmp_path / "out"), "--dosed", "S",
                 "--width", "4", "--depth", "1", "--steps", "6", "--restarts", "1"])
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
    parameters = pd.read_csv(tmp_path / "out" / "model" / "parameters.tsv", sep="\t").set_index("parameterId")
    assert float(parameters.loc["scale_Q", "nominalValue"]) == pytest.approx(2.0)
    assert not bool(parameters.loc["scale_Q", "estimate"])
    with h5py.File(tmp_path / "out" / "model" / "net1_ps.hdf5") as weights:
        assert weights["parameters/net1/layer1/weight"].shape == (4, 5)


def test_a_curve_of_checkpoints_imports_the_held_out_set_once(tmp_path, monkeypatch):
    """A budget curve scores every checkpoint against one held-out set. Preparing once must stage
    and import once, however many fields are then scored against it."""
    from petab_harness.neural_ode import train as T
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True))

    imported = []
    real = T._import
    monkeypatch.setattr(T, "_import", lambda *a, **k: (imported.append(1), real(*a, **k))[1])

    prepared = H.prepare(result, held_out, tmp_path / "curve", training=problem)
    assert not isinstance(prepared, dict), prepared
    payloads = [H.score_prepared(prepared, result.problem.model.nns, result.parameters)
                for _ in range(6)]

    assert len(imported) == 1, f"one import for six scorings, got {len(imported)}"
    assert all(p["status"] == "ok" and p["mode"] == "fixed" for p in payloads)
    # Same field six times must give the same score: the context carries no state between calls.
    assert len({round(p["nll"], 9) for p in payloads}) == 1
    # And the convenience wrapper still agrees with the prepared path.
    assert H.score(result, held_out, tmp_path / "once",
                   training=problem)["nll"] == pytest.approx(payloads[0]["nll"])


def test_a_prepared_set_refuses_a_transfer_as_score_does(tmp_path):
    """Preparing is not a way round the fixed-theta rule: a transfer set is refused up front."""
    from petab_mcp.validation_data import read_held_out

    problem, result = _trained(tmp_path)
    held_out = read_held_out(write_held_out(tmp_path / "validation", new_condition=True, transfer=True))
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

    perturbed = jtu.tree_map(lambda x: x * 1.5 if hasattr(x, "shape") else x, result.problem.model.nns)
    a = H.score_prepared(prepared, result.problem.model.nns, result.parameters)
    b = H.score_prepared(prepared, perturbed, result.parameters)
    assert a["status"] == "ok" and b["status"] == "ok"
    assert a["nll"] != pytest.approx(b["nll"]), "a changed field must move the held-out score"
