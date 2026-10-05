"""The TRANSFER kind of held-out check: a second problem on the same mechanism, on which both the
candidate and the reference are REFIT, from gemot's side.

The server's half (which tables move onto the candidate's view, how a bundle's units are read, the
registered tool, and the AMICI-backed ``score_session`` refit) is pinned in PEtab-MCP's
tests/test_validation_data_transfer.py. What is tested here, as in ``test_validation_data.py``, is
the assembly path around it: what the materialiser writes and refuses, and what the reference refit
and the judge do with a ``mode: transfer`` payload.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import validation_data as V

TOY = Path(__file__).parent / "data" / "toy_v1"


# ------------------------------------------------------------- the tables the tests build from
def _params(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "parameterId",
            "parameterScale",
            "lowerBound",
            "upperBound",
            "nominalValue",
            "estimate",
        ],
    )
    return df.set_index("parameterId")


def _conditions(ids, **cols):
    df = pd.DataFrame({"conditionId": list(ids), **{k: list(v) for k, v in cols.items()}})
    return df.set_index("conditionId")


def _rows(condition, times=(0.0, 1.0), observable="obs_B"):
    return pd.DataFrame(
        {
            "observableId": [observable] * len(times),
            "simulationConditionId": [condition] * len(times),
            "time": list(times),
            "measurement": [1.0] * len(times),
        }
    )


def _candidate():
    """A candidate problem stand-in with the toy problem's tables: A_init fixed at 1.0."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=_rows("c0"),
        observable_df=pd.DataFrame(
            {"observableFormula": ["B"], "noiseFormula": ["sigma_B"]},
            index=pd.Index(["obs_B"], name="observableId"),
        ),
        condition_df=_conditions(["c0"]),
        parameter_df=_params(
            [
                ("k1", "log10", 1e-4, 100.0, 0.5, 1),
                ("k2", "log10", 1e-4, 100.0, 0.1, 1),
                ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
                ("A_init", "lin", 0.0, 100.0, 1.0, 0),
            ]
        ),
    )


# ----------------------------------------------------------------- what the materialiser writes
def _mini_collection(
    tmp_path,
    *,
    held_sbml_same=True,
    extra_condition_column=False,
    redefine_observable=False,
    extra_observable=False,
    condition_name=False,
):
    """A two-problem collection built from the toy: the TRAINING problem verbatim, and a HELD-OUT
    problem on the same SBML with condition `c1`, its own parameter values (A_init = 2) and
    measurements on it; optionally with an extra observable `obs_A` (and its sigma) the training
    problem lacks, or a free-text `conditionName` column."""
    name = "Toy_Model"
    bdir = tmp_path / name
    bdir.mkdir()
    for f in (
        "conversion_reaction.xml",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
    ):
        shutil.copy(TOY / f, bdir / f)
    (bdir / f"{name}.yaml").write_text(
        "format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
        "- sbml_files: [conversion_reaction.xml]\n  condition_files: [conditions.tsv]\n"
        "  measurement_files: [measurements.tsv]\n  observable_files: [observables.tsv]\n"
    )

    held = bdir / "held"
    held.mkdir()
    sbml = "../conversion_reaction.xml"
    if not held_sbml_same:
        text = (
            (TOY / "conversion_reaction.xml")
            .read_text()
            .replace('id="conversion_reaction"', 'id="conversion_reaction_v2"')
        )
        (held / "other.xml").write_text(text)
        sbml = "other.xml"
    cond = pd.DataFrame({"conditionId": ["c1"]})
    if extra_condition_column:
        cond["dose"] = [1.0]
    if condition_name:
        cond["conditionName"] = ["the held-out system"]
    cond.to_csv(held / "conditions.tsv", sep="\t", index=False)
    obs = pd.read_csv(TOY / "observables.tsv", sep="\t")
    if redefine_observable:
        obs.loc[obs["observableId"] == "obs_B", "observableFormula"] = "2 * B"
    pars = pd.read_csv(TOY / "parameters.tsv", sep="\t")
    pars.loc[pars["parameterId"] == "A_init", "nominalValue"] = 2.0
    meas = pd.read_csv(TOY / "measurements.tsv", sep="\t").assign(simulationConditionId="c1")
    if extra_observable:
        obs = pd.concat(
            [
                obs,
                pd.DataFrame(
                    {
                        "observableId": ["obs_A"],
                        "observableFormula": ["A"],
                        "noiseFormula": ["sigma_A"],
                    }
                ),
            ],
            ignore_index=True,
        )
        pars = pd.concat(
            [
                pars,
                pd.DataFrame(
                    [
                        {
                            "parameterId": "sigma_A",
                            "parameterScale": "lin",
                            "lowerBound": 1e-4,
                            "upperBound": 100.0,
                            "nominalValue": 0.1,
                            "estimate": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        meas = pd.concat([meas, meas.assign(observableId="obs_A")], ignore_index=True)
    obs.to_csv(held / "observables.tsv", sep="\t", index=False)
    pars.to_csv(held / "parameters.tsv", sep="\t", index=False)
    meas.to_csv(held / "measurements.tsv", sep="\t", index=False)
    (held / "held.yaml").write_text(
        f"format_version: 1\nparameter_file: parameters.tsv\nproblems:\n"
        f"- sbml_files: [{sbml}]\n  condition_files: [conditions.tsv]\n"
        f"  measurement_files: [measurements.tsv]\n  observable_files: [observables.tsv]\n"
    )
    return name, tmp_path


@pytest.mark.parametrize("kind, needle", [("identical", "identical"), ("unknown", "k_new")])
def test_transfer_setup_rejects_invalid_source_parameters_before_pruning(tmp_path, kind, needle):
    """Compare against the full published table at setup, where reference kinetics are known.
    An unknown estimated source row must not disappear through the unused-parameter pruning."""
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path)
    bdir = coll / name
    path = bdir / "held/parameters.tsv"
    if kind == "identical":
        shutil.copy(bdir / "parameters.tsv", path)
    else:
        pars = pd.read_csv(path, sep="\t")
        extra = pars.loc[pars.parameterId == "k1"].assign(parameterId="k_new")
        pd.concat([pars, extra]).to_csv(path, sep="\t", index=False)
    published = benchmark._load_problem(name, coll, flatten=False)
    with pytest.raises(ValueError, match=needle):
        benchmark._transfer_problem(
            name=name,
            transfer_yaml="held/held.yaml",
            collection=coll,
            published=published,
            training_measurement_df=published.measurement_df,
        )


def test_the_transfer_problem_is_the_held_out_problem_verbatim(tmp_path):
    """Nothing is cropped: what the paper measured on the second system is what both sides are
    refit on, and the held-out problem's own tables (its `A_init`) are the start point. A
    `conditionName` column is free text, not an input, and does not make it a different model."""
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path, condition_name=True)
    published = benchmark._load_problem(name, coll, flatten=False)
    vprob = benchmark._transfer_problem(
        name=name,
        transfer_yaml="held/held.yaml",
        collection=coll,
        published=published,
        training_measurement_df=published.measurement_df,
    )
    assert list(vprob.observable_df.index) == ["obs_B"]
    assert set(vprob.measurement_df["simulationConditionId"]) == {"c1"}
    assert len(vprob.measurement_df) == 6
    assert vprob.parameter_df.loc["A_init", "nominalValue"] == 2.0


@pytest.mark.parametrize(
    "knob, needle",
    [
        ({"held_sbml_same": False}, "SBML differs"),
        ({"extra_condition_column": True}, "condition table has columns"),
        ({"redefine_observable": True}, "defined differently"),
        ({"extra_observable": True}, "obs_A"),
    ],
)
def test_a_held_out_problem_that_is_not_the_same_mechanism_is_refused(tmp_path, knob, needle):
    """'Same mechanism, different system' is the premise of the comparison. A variant that edited
    the model, added an input, or redefined an observable would refit a DIFFERENT structure as the
    bar; one that measures a readout the training problem never defined (Bahia's hospital and ICU
    beds) cannot be scored on it, and dropping the readout would score a subset of the experiment
    without saying so. All refused at setup, where the message can say what changed."""
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path, **knob)
    published = benchmark._load_problem(name, coll, flatten=False)
    with pytest.raises(ValueError, match=needle):
        benchmark._transfer_problem(
            name=name,
            transfer_yaml="held/held.yaml",
            collection=coll,
            published=published,
            training_measurement_df=published.measurement_df,
        )


@pytest.mark.parametrize("rename", [False, True])
def test_materialize_writes_the_three_tables_beside_problem_and_the_server_reads_a_transfer(
    tmp_path, rename
):
    """Read back the transfer bundle outside problem/, with matching IDs after relabelling.
    Its parameter table must select transfer mode and retain the reference's parameter rows."""
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path)
    example_dir = tmp_path / "example"
    out = benchmark.materialize(
        name=name,
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=example_dir,
        transfer_yaml="held/held.yaml",
        rename_ids={"c1": "held_out", "A_init": "seed"} if rename else None,
        collection=coll,
    )
    assert out["validation_measurements"] == 6
    vpath = benchmark.validation_measurements_path(example_dir)
    assert vpath.parent.name == benchmark.VALIDATION_DIR
    assert not (example_dir / "problem" / "measurements.tsv").read_text().count("c1")
    held = V.read_held_out(vpath)
    assert held.mode == V.MODE_TRANSFER
    condition = "held_out" if rename else "c1"
    assert list(held.condition_df.index) == [condition] and "conditionName" not in held.condition_df
    assert set(held.measurement_df["simulationConditionId"]) == {condition}
    assert held.parameter_df.loc["seed" if rename else "A_init", "nominalValue"] == 2.0
    assert "k1" in held.parameter_df.index, "verbatim: which rows apply is the scorer's rule"


def test_the_two_kinds_of_held_out_set_are_mutually_exclusive(tmp_path):
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        benchmark.materialize(
            name=name,
            shipped_param_ids=["sigma_B"],
            example_dir=tmp_path / "e",
            validation_yaml="held/held.yaml",
            transfer_yaml="held/held.yaml",
            collection=coll,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        benchmark.write_reference_for(
            name=name,
            example_dir=tmp_path / "e",
            collection=coll,
            validation_yaml="held/held.yaml",
            transfer_yaml="held/held.yaml",
        )


@pytest.mark.parametrize("fixed", [False, True])
def test_materialize_retires_a_transfer_bundle_when_its_declaration_changes(tmp_path, fixed):
    """An old parameters.tsv must not select transfer mode for new fixed-θ data; removing the
    declaration must also remove the measurements that trigger the judge's held-out call."""
    from gemot import benchmark

    name, coll = _mini_collection(tmp_path)
    bdir = coll / name
    pd.read_csv(bdir / "measurements.tsv", sep="\t").assign(time=lambda df: df.time + 100).to_csv(
        bdir / "fixed_measurements.tsv", sep="\t", index=False
    )
    (bdir / "fixed.yaml").write_text(
        (bdir / f"{name}.yaml").read_text().replace("measurements.tsv", "fixed_measurements.tsv")
    )
    kwargs = dict(
        name=name,
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=tmp_path / "example",
        collection=coll,
    )
    benchmark.materialize(**kwargs, transfer_yaml="held/held.yaml")
    benchmark.materialize(**kwargs, validation_yaml="fixed.yaml" if fixed else None)
    path = benchmark.validation_measurements_path(kwargs["example_dir"])
    assert not path.with_name(V.HELD_OUT_CONDITIONS).exists()
    assert not path.with_name(V.HELD_OUT_PARAMETERS).exists()
    assert path.exists() == fixed
    if fixed:
        assert V.read_held_out(path).mode == V.MODE_FIXED


def test_the_reference_refit_lands_in_reference_json_with_its_mode_and_a_failure_is_recorded(
    tmp_path, monkeypatch
):
    """`write_reference_for` has two arms (published split / freed split) and one tail; the transfer
    block has to reach the file from the tail, tagged `mode: transfer`, or the judge would subtract a
    fixed-θ bar from a transfer payload. And a refit that raises must be RECORDED as an error, not
    allowed to discard the training bar that was just fit."""
    from gemot import benchmark, simulate

    name, coll = _mini_collection(tmp_path)
    example_dir = tmp_path / "example"
    (example_dir / "problem").mkdir(parents=True)

    def _fake_training(problem, *, path, **_):
        ref = {"nll": 1.0, "k": 2, "n": 6}
        Path(path).write_text(json.dumps(ref))
        return ref

    seen = {}

    # No budget parameters: the training knob passed below must NOT reach the transfer refit, and
    # this signature is what makes that a failure rather than a silently-accepted keyword.
    def _fake_refit(problem, *, build_dir, model_name, n_starts, units=None):
        seen["n_starts"] = n_starts
        seen["k"] = int((problem.parameter_df["estimate"].astype(float) == 1).sum())
        seen["A_init"] = float(problem.parameter_df.loc["A_init", "nominalValue"])
        seen["build_dir"] = str(build_dir)
        return {"status": "ok", "mode": "transfer", "nll": 3.0, "k": seen["k"], "n": 6}

    monkeypatch.setattr(simulate, "write_reference_from_problem", _fake_training)
    monkeypatch.setattr(V, "refit_problem", _fake_refit)
    ref = benchmark.write_reference_for(
        name=name,
        example_dir=example_dir,
        collection=coll,
        n_starts=7,
        per_start_timeout=11.0,
        transfer_yaml="held/held.yaml",
    )
    on_disk = json.loads((example_dir / "problem" / "reference.json").read_text())
    assert on_disk == ref
    assert ref["validation"] == {"status": "ok", "mode": "transfer", "nll": 3.0, "k": 2, "n": 6}
    assert seen["n_starts"] == 7
    assert seen["A_init"] == 2.0, "the reference refits the HELD-OUT problem, at its own values"
    assert "pristine_transfer_" in seen["build_dir"]

    monkeypatch.setattr(V, "refit_problem", lambda *a, **k: {"status": "non_finite"})
    ref = benchmark.write_reference_for(
        name=name, example_dir=example_dir, collection=coll, transfer_yaml="held/held.yaml"
    )
    assert ref["validation"] == {"status": "non_finite", "mode": "transfer"}

    def _raising(*a, **k):
        raise RuntimeError("no module")

    monkeypatch.setattr(V, "refit_problem", _raising)
    ref = benchmark.write_reference_for(
        name=name, example_dir=example_dir, collection=coll, transfer_yaml="held/held.yaml"
    )
    assert ref["nll"] == 1.0, "the training bar survives"
    assert ref["validation"]["status"] == "error" and ref["validation"]["mode"] == "transfer"
    assert "no module" in ref["validation"]["reason"]


# ------------------------------------------ a curated set that ships its own parameter table
def _toy_published():
    import petab.v1 as petab

    return petab.Problem.from_yaml(str(TOY / "problem.yaml"))


def _curated_transfer(tmp_path, *, delta=None, conditions=("c1",)):
    """A curated set beside the toy: measurements on `conditions`, a conditions.tsv naming them and
    a parameters.tsv holding the `delta` rows — header-only when there are none."""
    d = tmp_path / "curated"
    d.mkdir(exist_ok=True)
    pd.concat([_rows(c) for c in conditions]).to_csv(d / "measurements.tsv", sep="\t", index=False)
    _conditions(conditions).reset_index().to_csv(d / V.HELD_OUT_CONDITIONS, sep="\t", index=False)
    _params(delta or []).reset_index().to_csv(d / V.HELD_OUT_PARAMETERS, sep="\t", index=False)
    return "curated/measurements.tsv"


def test_a_curated_table_with_a_parameter_table_beside_it_is_a_transfer_set(tmp_path):
    """The file holds only what differs; the published table is the rest. Here `A_init` moves to 2
    and every other row, kinetics included, is the published one — the start of the reference
    refit — and the parameter table's presence is what tells the server to refit, not evaluate.
    The set's condition table REPLACES the training one rather than being appended to it."""
    from gemot import benchmark

    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
    )

    published = _toy_published().parameter_df
    assert held.mode == V.MODE_TRANSFER
    assert list(held.condition_df.index) == ["c1"]
    assert held.parameter_df.loc["A_init", "nominalValue"] == 2.0
    assert set(held.parameter_df.index) == set(published.index)
    assert held.parameter_df.loc["k1", "nominalValue"] == published.loc["k1", "nominalValue"]
    assert list(held.parameter_df["estimate"]) == list(published["estimate"])


@pytest.mark.parametrize(
    "delta, conditions, needle",
    [
        ([], ("c0",), "identical"),
        ([("k_new", "log10", 1e-3, 1e3, 1.0, 1)], ("c1",), "k_new"),
    ],
)
def test_a_curated_parameter_table_is_checked_like_a_collection_transfer_problem(
    tmp_path, delta, conditions, needle
):
    """A header-only file on the published conditions changes nothing, so nothing transfers; a new
    estimated mechanism parameter is a rate the published model cannot carry. The same rule as for
    a collection's second problem, at setup, naming the file."""
    from gemot import benchmark

    table = _curated_transfer(tmp_path, delta=delta, conditions=conditions)

    with pytest.raises(ValueError, match=needle) as exc:
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )
    assert V.HELD_OUT_PARAMETERS in str(exc.value)


def test_a_curated_transfer_condition_table_must_keep_the_published_columns(tmp_path):
    from gemot import benchmark

    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])
    pd.DataFrame({"conditionId": ["c1"], "dose": [1.0]}).to_csv(
        tmp_path / "curated" / V.HELD_OUT_CONDITIONS, sep="\t", index=False
    )

    with pytest.raises(ValueError, match="different models"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )


def test_a_bound_only_delta_is_a_change(tmp_path):
    """Identity is judged on every column: a delta that only moves `A_init`'s upper bound is a
    different problem, not "identical"."""
    from gemot import benchmark

    table = _curated_transfer(tmp_path)
    pd.DataFrame({"parameterId": ["A_init"], "upperBound": [50.0]}).to_csv(
        tmp_path / "curated" / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
    )

    assert held.mode == V.MODE_TRANSFER
    assert held.parameter_df.loc["A_init", "upperBound"] == 50.0
    assert (
        held.parameter_df.loc["A_init", "nominalValue"]
        == _toy_published().parameter_df.loc["A_init", "nominalValue"]
    )


def test_new_input_values_make_a_system_and_a_pruned_row_does_not(tmp_path):
    """lucarelli's shape, on the toy: `k1` as a per-condition input the published table sets to
    itself, the curated table to a number. Header-only delta. The number is what makes it another
    system, and the retired `k1` row leaves the table. The same files with the published value on
    the published condition are the published problem again — refused as identical even though
    pruning `k1` would have made the two parameter tables differ."""
    from gemot import benchmark

    published = _toy_published()
    published.condition_df["k1"] = ["k1"]
    table = _curated_transfer(tmp_path, conditions=("c1",))
    cpath = tmp_path / "curated" / V.HELD_OUT_CONDITIONS
    pd.DataFrame({"conditionId": ["c1"], "k1": [0.5]}).to_csv(cpath, sep="\t", index=False)

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=published
    )
    assert held.mode == V.MODE_TRANSFER and "k1" not in held.parameter_df.index

    _curated_transfer(tmp_path, conditions=("c0",))
    pd.DataFrame({"conditionId": ["c0"], "k1": ["k1"]}).to_csv(cpath, sep="\t", index=False)
    with pytest.raises(ValueError, match="identical"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=published
        )


def test_a_new_experiment_is_a_system_without_new_inputs_or_a_delta(tmp_path):
    """crauste's shape: no input columns, no parameter change, two conditions the published table
    never had — another experiment on the same mechanism, every rate refit. Two deliberate files,
    so it is not a mis-annotation, and the identity check stands aside."""
    from gemot import benchmark

    table = _curated_transfer(tmp_path, conditions=("F5_tumour", "F5_virus"))

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
    )

    assert held.mode == V.MODE_TRANSFER
    assert list(held.condition_df.index) == ["F5_tumour", "F5_virus"]
    assert held.parameter_df.equals(_toy_published().parameter_df)


def test_a_sigma_no_held_out_row_informs_is_not_a_parameter_of_the_refit(tmp_path):
    """The set measures `obs_B` only: `sigma_A` leaves the transfer table (it would price `k` for
    a parameter nothing informs), and a delta row for it names nothing the set uses."""
    from gemot import benchmark

    published = _toy_published()
    published.observable_df = pd.concat(
        [
            published.observable_df,
            pd.DataFrame(
                {"observableFormula": ["A"], "noiseFormula": ["sigma_A"]},
                index=pd.Index(["obs_A"], name="observableId"),
            ).reindex(columns=published.observable_df.columns),
        ]
    )
    published.parameter_df = pd.concat(
        [
            published.parameter_df,
            _params([("sigma_A", "lin", 1e-4, 100.0, 0.1, 1)]).reindex(
                columns=published.parameter_df.columns
            ),
        ]
    )
    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=published
    )
    assert "sigma_A" not in held.parameter_df.index and "sigma_B" in held.parameter_df.index

    pd.DataFrame({"parameterId": ["sigma_A"], "upperBound": [5.0]}).to_csv(
        tmp_path / "curated" / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="sigma_A"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=published
        )


def test_an_empty_curated_table_is_refused_at_setup(tmp_path):
    from gemot import benchmark

    table = _curated_transfer(tmp_path)
    mpath = tmp_path / "curated" / "measurements.tsv"
    pd.read_csv(mpath, sep="\t").iloc[0:0].to_csv(mpath, sep="\t", index=False)

    with pytest.raises(ValueError, match="no measurements"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )


def test_materialize_writes_a_curated_transfer_bundle_the_server_reads_as_transfer(
    tmp_path, monkeypatch
):
    """On disk the curated set is indistinguishable from a collection transfer bundle: the FULL
    parameter table beside the measurements, which `read_held_out` turns into `mode: transfer`."""
    from gemot import benchmark

    monkeypatch.setattr(benchmark, "_load_problem", lambda *a, **k: _toy_published())
    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])

    out = benchmark.materialize(
        name="X",
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=tmp_path,
        validation_table=table,
    )
    held = V.read_held_out(benchmark.validation_measurements_path(tmp_path))

    assert out["validation_measurements"] == 2 and out["validation_conditions"] == 1
    assert held.mode == V.MODE_TRANSFER
    assert list(held.condition_df.index) == ["c1"]
    assert held.parameter_df.loc["A_init", "nominalValue"] == 2.0
    assert "k1" in held.parameter_df.index


def test_the_reference_refits_a_curated_transfer_set_from_the_published_point(
    tmp_path, monkeypatch
):
    """`write_reference_for` routes a curated transfer set to `_transfer_reference`: the training
    refit is handed NO held-out table (a transfer set is not scored at the fitted point), and the
    refit starts from the published table with the set's own rows laid over it, on the set's own
    conditions."""
    from gemot import benchmark, simulate

    monkeypatch.setattr(benchmark, "_load_problem", lambda *a, **k: _toy_published())
    (tmp_path / "problem").mkdir()
    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])
    seen = {}

    def _fake_training(problem, *, path, held_out=None, **_):
        seen["training_held_out"] = held_out
        ref = {"nll": 1.0, "k": 2, "n": 6}
        Path(path).write_text(json.dumps(ref))
        return ref

    def _fake_refit(problem, *, build_dir, model_name, n_starts, units=None):
        seen["A_init"] = float(problem.parameter_df.loc["A_init", "nominalValue"])
        seen["conditions"] = list(problem.condition_df.index)
        seen["n_starts"] = n_starts
        return {"status": "ok", "mode": "transfer", "nll": 3.0, "k": 2, "n": 2}

    monkeypatch.setattr(simulate, "write_reference_from_problem", _fake_training)
    monkeypatch.setattr(V, "refit_problem", _fake_refit)

    ref = benchmark.write_reference_for(
        name="X", example_dir=tmp_path, validation_table=table, n_starts=3
    )

    assert seen["training_held_out"] is None
    assert seen["A_init"] == 2.0 and seen["conditions"] == ["c1"] and seen["n_starts"] == 3
    assert ref["validation"]["mode"] == V.MODE_TRANSFER and ref["nll"] == 1.0
    assert json.loads((tmp_path / "problem" / "reference.json").read_text()) == ref


def test_the_fixed_point_reference_scorer_refuses_a_transfer_set(tmp_path):
    """The guard behind the routing above: handing the training refit a transfer set would score
    it at the reference's fitted point and record an `ok` fixed-mode bar for data that is refit."""
    from gemot import simulate

    held = V.HeldOut(_rows("c1"), parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]))

    with pytest.raises(ValueError, match="REFIT"):
        simulate.write_reference_from_problem(
            _toy_published(),
            build_dir=tmp_path,
            model_name="m",
            path=tmp_path / "r.json",
            held_out=held,
        )


# ------------------------------------------------------ the systems a transfer set is refit on
def test_a_curated_transfer_set_names_the_systems_it_is_refit_on(tmp_path):
    """`units.tsv` beside a curated table reaches the HeldOut; a measured condition it leaves out,
    or a units file beside a fixed-θ set (nothing is refit there), is a setup error naming it."""
    from gemot import benchmark

    table = _curated_transfer(
        tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)], conditions=("c1", "c2")
    )
    upath = tmp_path / "curated" / V.HELD_OUT_UNITS
    pd.DataFrame({"conditionId": ["c1", "c2"], "unitId": ["u1", "u2"]}).to_csv(
        upath, sep="\t", index=False
    )

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
    )
    assert held.units == {"c1": "u1", "c2": "u2"}

    pd.DataFrame({"conditionId": ["c1"], "unitId": ["u1"]}).to_csv(upath, sep="\t", index=False)
    with pytest.raises(ValueError, match="c2"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )

    (tmp_path / "curated" / V.HELD_OUT_PARAMETERS).unlink()
    with pytest.raises(ValueError, match="fitted point"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )


def test_materialize_writes_the_units_beside_the_bundle_and_the_reference_refits_by_them(
    tmp_path, monkeypatch
):
    from gemot import benchmark, simulate

    monkeypatch.setattr(benchmark, "_load_problem", lambda *a, **k: _toy_published())
    (tmp_path / "problem").mkdir()
    table = _curated_transfer(
        tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)], conditions=("c1", "c2")
    )
    pd.DataFrame({"conditionId": ["c1", "c2"], "unitId": ["u1", "u2"]}).to_csv(
        tmp_path / "curated" / V.HELD_OUT_UNITS, sep="\t", index=False
    )

    benchmark.materialize(
        name="X",
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=tmp_path,
        validation_table=table,
    )
    held = V.read_held_out(benchmark.validation_measurements_path(tmp_path))
    assert held.units == {"c1": "u1", "c2": "u2"}

    seen = {}

    def _fake_refit(problem, *, build_dir, model_name, n_starts, units=None):
        seen["units"] = units
        return {"status": "ok", "mode": "transfer", "nll": 3.0, "k": 2, "n": 4}

    monkeypatch.setattr(
        simulate,
        "write_reference_from_problem",
        lambda problem, *, path, **_: (Path(path).write_text("{}"), {"nll": 1.0, "k": 2, "n": 6})[
            1
        ],
    )
    monkeypatch.setattr(V, "refit_problem", _fake_refit)
    benchmark.write_reference_for(name="X", example_dir=tmp_path, validation_table=table)
    assert seen["units"] == {"c1": "u1", "c2": "u2"}


def test_an_empty_units_declaration_is_incomplete_not_absent(tmp_path, monkeypatch):
    """A header-only units.tsv is `{}`, and `{}` maps no condition: refused at score time and at
    setup, rather than read as "one system"."""
    import pypesto.petab

    from gemot import benchmark

    monkeypatch.setattr(pypesto.petab, "PetabImporter", lambda *a, **k: None)
    (tmp_path / "u.tsv").write_text("conditionId\tunitId\n")
    assert V.read_units(tmp_path / "u.tsv") == {}
    problem = _candidate()
    out = V.refit_problem(problem, build_dir="/tmp/x", model_name="m", n_starts=1, units={})
    assert out["status"] == V.STATUS_TABLE_MISMATCH and "c0" in out["reason"]

    table = _curated_transfer(tmp_path, delta=[("A_init", "lin", 0.0, 100.0, 2.0, 0)])
    (tmp_path / "curated" / V.HELD_OUT_UNITS).write_text("conditionId\tunitId\n")
    with pytest.raises(ValueError, match="c1"):
        benchmark._curated_held_out(
            name="X", validation_table=table, example_dir=tmp_path, published=_toy_published()
        )
