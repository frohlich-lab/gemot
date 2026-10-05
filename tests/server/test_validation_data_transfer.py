"""The TRANSFER kind of held-out check: a second problem on the same mechanism, on which both the
candidate and the reference are REFIT (``petab_mcp.validation_data`` and its two consumers).

Tested as three surfaces, apart from each other, exactly as the fixed-θ kind is in
``test_validation_data.py`` — the assembly path is where the defects live: which tables move onto
the candidate's view and which do not, what the materialiser writes and refuses, and what the judge
and the history CSV do with a ``mode: transfer`` payload. The one AMICI-backed test drives the whole
server-side path (``score_session``) on the toy problem, because "the refit runs and returns the
shape the judge reads" is not something a mocked importer can establish.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import validation_data as V

TOY = Path(__file__).parents[1] / "data" / "toy_v1"


# ------------------------------------------------------------------ what makes a set a transfer
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


def test_a_held_out_set_is_a_transfer_set_iff_it_brings_its_own_parameter_table(tmp_path):
    """The mode is READ OFF the files, not declared beside them: a parameter table is the statement
    that the parameters are not shared, and a flag next to it would be a second thing able to
    disagree with the first."""
    meas = tmp_path / "measurements.tsv"
    _rows("c1").to_csv(meas, sep="\t", index=False)
    assert V.read_held_out(meas).mode == V.MODE_FIXED

    _conditions(["c1"]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_CONDITIONS, sep="\t", index=False
    )
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_FIXED and list(held.condition_df.index) == ["c1"]

    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_TRANSFER
    assert held.parameter_df.loc["A_init", "nominalValue"] == 2.0


# ------------------------------------------------------------- moving the candidate onto the set
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


_SHIPPED = _params(
    [
        ("k1", "log10", 1e-4, 100.0, 0.5, 1),
        ("k2", "log10", 1e-4, 100.0, 0.1, 1),
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
        ("A_init", "lin", 0.0, 100.0, 1.0, 0),
    ]
)

_HELD_OUT_PARAMS = _params(
    [
        ("k1", "log10", 1e-4, 100.0, 7.0, 1),  # estimated: the published held-out optimum
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),  # fixed, unchanged
        ("A_init", "lin", 0.0, 100.0, 2.0, 0),  # fixed, CHANGED: the input that must move
        ("sigma_extra", "lin", 1e-4, 100.0, 3.0, 0),  # fixed, never in the training problem
    ]
)


def test_the_view_moves_only_the_changed_fixed_inputs_and_never_the_start_point():
    """Three kinds of held-out row and three different fates. `A_init` (fixed, changed) is written
    into the candidate; `k1` (estimated) is NOT — the held-out table's value is the published optimum
    for that system and would hand the candidate the answer, so its refit starts from its own
    training value; `sigma_extra` (a fixed quantity the training problem never named) binds to
    nothing. And the caller's problem is untouched throughout."""
    problem = _candidate()
    held = V.HeldOut(
        measurement_df=_rows("c1"), condition_df=_conditions(["c1"]), parameter_df=_HELD_OUT_PARAMS
    )
    view, failed = V.transfer_view(problem, held, _SHIPPED)
    assert failed is None
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    assert view.parameter_df.loc["k1", "nominalValue"] == 0.5, (
        "the start point is the candidate's own"
    )
    assert "sigma_extra" not in view.parameter_df.index
    assert list(view.condition_df.index) == ["c1"]
    assert list(view.measurement_df["simulationConditionId"]) == ["c1", "c1"]
    assert view.visualization_df is None
    # nothing leaked back
    assert problem.parameter_df.loc["A_init", "nominalValue"] == 1.0
    assert list(problem.condition_df.index) == ["c0"]
    assert list(problem.measurement_df["simulationConditionId"]) == ["c0", "c0"]


def test_a_candidate_that_cannot_receive_a_changed_input_is_a_finding_not_a_crash():
    """oliveira's `population` changes between the interior and the whole state. A model that
    hard-coded it (no row), renamed it, or freed it has no fixed parameter to receive the new value,
    so it cannot be moved to the held-out system at all — reported as its own status, because a
    refit at the TRAINING population would be a plausible, wrong number."""
    held = V.HeldOut(
        measurement_df=_rows("c1"), condition_df=_conditions(["c1"]), parameter_df=_HELD_OUT_PARAMS
    )

    no_row = _candidate()
    no_row.parameter_df = no_row.parameter_df.drop(index="A_init")
    view, failed = V.transfer_view(no_row, held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING
    assert "A_init" in failed["reason"]

    freed = _candidate()
    freed.parameter_df.loc["A_init", "estimate"] = 1
    view, failed = V.transfer_view(freed, held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING


@pytest.mark.parametrize("rate", ["k1", "candidate_rate"])
def test_zero_shot_transfer_keeps_the_candidates_own_kinetic_parameters(rate):
    """The shipped subset omits kinetics; a candidate may also rename them. Neither permits
    copying the reference's fitted rates into the candidate or rejecting its transfer refit."""
    problem = _candidate()
    problem.parameter_df = problem.parameter_df.rename(index={"k1": rate})
    held = V.HeldOut(
        measurement_df=_rows("c1"), condition_df=_conditions(["c1"]), parameter_df=_HELD_OUT_PARAMS
    )
    view, failed = V.transfer_view(problem, held, _SHIPPED.drop(index=["k1", "k2"]))
    assert failed is None
    pd.testing.assert_frame_equal(
        view.parameter_df.drop(index="A_init"), problem.parameter_df.drop(index="A_init")
    )
    assert view.parameter_df.loc[rate, "nominalValue"] == 0.5
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0


def test_a_held_out_system_may_bring_its_own_estimated_noise_parameters():
    """A new noise sigma or observable scaling IS manageable: it enters through the observation
    model — the held-out measurement table's `noiseParameters` override — not the ODE."""
    meas = _rows("c1").assign(noiseParameters="sigma_new")
    rows = [
        ("k1", "log10", 1e-4, 100.0, 7.0, 1),
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
        ("A_init", "lin", 0.0, 100.0, 2.0, 0),
        ("sigma_new", "lin", 1e-4, 100.0, 0.3, 1),
    ]
    held = V.HeldOut(
        measurement_df=meas, condition_df=_conditions(["c1"]), parameter_df=_params(rows)
    )
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert failed is None, failed


def test_a_held_out_condition_table_with_other_columns_is_another_model():
    """A condition column is an input the model declares and the compiled module maps; a held-out
    table adding one describes a model the candidate is not, and reusing the module would be wrong."""
    held = V.HeldOut(
        measurement_df=_rows("c1"),
        condition_df=_conditions(["c1"], dose=[1.0]),
        parameter_df=_HELD_OUT_PARAMS,
    )
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_TABLE_MISMATCH
    assert "dose" in failed["reason"]


def test_held_out_conditions_are_checked_against_the_held_out_table_not_the_training_one():
    """The whole point of the condition swap: `c1` is not a training condition and must still pass,
    while a measurement row naming a condition in NEITHER table must not."""
    ok = V.HeldOut(
        measurement_df=_rows("c1"), condition_df=_conditions(["c1"]), parameter_df=_HELD_OUT_PARAMS
    )
    view, failed = V.transfer_view(_candidate(), ok, _SHIPPED)
    assert failed is None and view is not None

    bad = V.HeldOut(
        measurement_df=_rows("c_nowhere"),
        condition_df=_conditions(["c1"]),
        parameter_df=_HELD_OUT_PARAMS,
    )
    view, failed = V.transfer_view(_candidate(), bad, _SHIPPED)
    assert failed["status"] == V.STATUS_TABLE_MISMATCH and "c_nowhere" in failed["reason"]


def test_refit_declines_a_bad_table_before_any_build(monkeypatch):
    """Same contract as `score_problem`: the guards run before the importer, so an empty or foreign
    table costs nothing and names its own problem."""
    import pypesto.petab

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _candidate()
    problem.measurement_df = problem.measurement_df.iloc[0:0]
    out = V.refit_problem(problem, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_NO_DATA

    problem = _candidate()
    problem.measurement_df = _rows("c_elsewhere")
    out = V.refit_problem(problem, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_TABLE_MISMATCH and "c_elsewhere" in out["reason"]


# ------------------------------------------------ the whole server-side path, with AMICI
def test_score_session_refits_the_toy_on_a_transfer_set_and_leaves_the_session_alone(
    session, engine
):
    """The one AMICI-backed test: the toy model, its A_init doubled on a new condition `c1`, is
    REFIT on measurements it never saw. Pins the payload SHAPE the judge reads (`mode`, `k`, `n`,
    a finite `nll`), that `k` is the count of estimated rows (k1, k2 — not sigma_B or A_init), that
    the refit's optimum is no worse than its start point, and that the session's own table and
    model are exactly as they were: a transfer refit that wrote its optimum back would change what
    the judge scores on the training data."""

    meas = pd.read_csv(TOY / "measurements.tsv", sep="\t")
    held_meas = meas.assign(simulationConditionId="c1", measurement=meas["measurement"] * 2.0)
    held = V.HeldOut(
        measurement_df=held_meas,
        condition_df=_conditions(["c1"]),
        parameter_df=_params(
            [("A_init", "lin", 0.0, 100.0, 2.0, 0), ("k1", "log10", 1e-4, 100.0, 0.9, 1)]
        ),
    )
    before = session.problem.parameter_df.copy(deep=True)
    sbml_before = session.model_sbml()
    session.shipped_parameter_df = session.shipped_parameter_df.drop(index=["k1", "k2"])

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_TRANSFER
    assert out["k"] == 2 and out["n"] == len(held_meas) and out["n_conditions"] == 1
    assert out["observables"] == ["obs_B"] and out["n_starts"] == 2
    assert math.isfinite(out["nll"]) and out["chi2"] is not None
    # the doubled data at A_init = 2 is the SAME curve shape, so the refit from the training
    # optimum must land near the training likelihood — a refit at the training A_init (the
    # input NOT moved) could not
    view, _ = V.transfer_view(session.problem, held, session.shipped_parameter_df)
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    pd.testing.assert_frame_equal(session.problem.parameter_df, before)
    assert session.model_sbml() == sbml_before
    assert list(session.problem.condition_df.index) == ["c0"]


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


def test_the_server_tool_takes_n_starts_and_refits_a_transfer_bundle(tmp_path):
    """The registered tool is what the judge reaches over MCP: it must accept `n_starts` (or the
    judge's call fails and the run records `unavailable`) and must read the bundle as a transfer
    set. Driven in-process through `create_server` like the doc-consistency tests."""
    import asyncio

    from petab_mcp.server import create_server

    prob = tmp_path / "problem"
    shutil.copytree(TOY, prob)
    val = tmp_path / "validation"
    val.mkdir()
    pd.read_csv(TOY / "measurements.tsv", sep="\t").assign(simulationConditionId="c1").to_csv(
        val / "measurements.tsv", sep="\t", index=False
    )
    _conditions(["c1"]).reset_index().to_csv(val / V.HELD_OUT_CONDITIONS, sep="\t", index=False)
    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        val / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )

    mcp = create_server(
        str(prob / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        build_root=str(tmp_path / "build"),
        auto_fit=False,
        validation_measurements=str(val / "measurements.tsv"),
    )
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "n_starts" in tools["evaluate_validation_data"].parameters["properties"]


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
def test_read_held_out_reads_the_sets_units(tmp_path):
    """`units.tsv` beside the bundle: {conditionId: unitId}, the file the harness writes and the
    server reads, so the two agree on the split through the file and nowhere else."""
    _rows("c1").to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )
    pd.DataFrame({"conditionId": ["c1", "c2"], "unitId": ["u1", "u2"]}).to_csv(
        tmp_path / V.HELD_OUT_UNITS, sep="\t", index=False
    )

    held = V.read_held_out(tmp_path / "measurements.tsv")

    assert held.mode == V.MODE_TRANSFER and held.units == {"c1": "u1", "c2": "u2"}


def test_a_row_whose_condition_is_in_no_unit_is_declined_before_any_build(monkeypatch):
    import pypesto.petab

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _candidate()
    problem.measurement_df = pd.concat([_rows("c0"), _rows("c1")])
    problem.condition_df = _conditions(["c0", "c1"])

    out = V.refit_problem(
        problem, build_dir="/tmp/x", model_name="m", n_starts=1, units={"c0": "u0"}
    )

    assert out["status"] == V.STATUS_TABLE_MISMATCH and "c1" in out["reason"]


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


def test_score_session_refits_each_unit_on_its_own_and_sums_the_units(session, engine):
    """AMICI-backed: the toy's measurements on two conditions, two units. Each unit is refit alone
    from the same start, the payload sums nll and n over them and carries each unit's block, and
    `k` is one refit's count — not the sum, which would price the same rows twice."""
    meas = pd.read_csv(TOY / "measurements.tsv", sep="\t")
    held_meas = pd.concat(
        [
            meas.assign(simulationConditionId="c1", measurement=meas.measurement * 2),
            meas.assign(simulationConditionId="c2", measurement=meas.measurement * 3),
        ],
        ignore_index=True,
    )
    held = V.HeldOut(
        measurement_df=held_meas,
        condition_df=_conditions(["c1", "c2"]),
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]),
        units={"c1": "u1", "c2": "u2"},
    )

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_TRANSFER and set(out["units"]) == {"u1", "u2"}
    assert out["n"] == len(held_meas) == sum(u["n"] for u in out["units"].values())
    assert math.isclose(out["nll"], sum(u["nll"] for u in out["units"].values()))
    assert out["units"]["u1"]["k"] == 2 and out["k"] == 4, "k sums the units' own counts"
    assert all(u["n_conditions"] == 1 and "fitted" in u for u in out["units"].values())
    assert list(session.problem.condition_df.index) == ["c0"]


def test_a_single_named_unit_is_still_carried_under_units(monkeypatch):
    """`units` names the systems; naming one is a declaration, not the absence of one, so its
    block is carried the same way — a consumer reading `units` must not find it missing."""
    seen = []

    def _fake_one(view, *, build_dir, model_name, n_starts):
        seen.append(view)
        return {
            "status": "ok",
            "mode": "transfer",
            "nll": 1.0,
            "k": 1,
            "n": 2,
            "chi2": 2.0,
            "n_conditions": 1,
            "observables": ["obs_B"],
            "n_starts": n_starts,
            "n_converged": 1,
            "fitted": {},
        }

    monkeypatch.setattr(V, "_refit_one", _fake_one)
    problem = _candidate()
    problem.measurement_df = _rows("c0")

    out = V.refit_problem(
        problem, build_dir="/tmp/x", model_name="m", n_starts=1, units={"c0": "only"}
    )

    assert list(out["units"]) == ["only"] and out["nll"] == 1.0 and out["k"] == 1
    assert len(seen) == 1


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


def test_read_units_refuses_a_condition_in_two_units_or_a_blank(tmp_path):
    (tmp_path / "u.tsv").write_text("conditionId\tunitId\nc1\tu1\nc1\tu2\n")
    with pytest.raises(ValueError, match="c1"):
        V.read_units(tmp_path / "u.tsv")
    (tmp_path / "u.tsv").write_text("conditionId\tunitId\nc1\t\n")
    with pytest.raises(ValueError, match="unitId"):
        V.read_units(tmp_path / "u.tsv")


def test_a_refit_sees_only_the_observables_its_rows_measure():
    """The per-refit view keeps the observables the rows measure and nothing else — the sigma of
    an unmeasured observable is not a parameter of that refit (pruned by petab's lint where a model
    is present)."""
    problem = _candidate()
    problem.observable_df = pd.concat(
        [
            problem.observable_df,
            pd.DataFrame(
                {"observableFormula": ["A"], "noiseFormula": ["1"]},
                index=pd.Index(["obs_A"], name="observableId"),
            ).reindex(columns=problem.observable_df.columns),
        ]
    )

    uview = V._unit_view(problem, _rows("c0"))

    assert list(uview.observable_df.index) == ["obs_B"]
    assert list(problem.observable_df.index) == ["obs_B", "obs_A"], "the caller's view is untouched"
