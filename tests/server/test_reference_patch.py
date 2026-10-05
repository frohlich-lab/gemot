"""The reference routing for a held-out set when an example STATES its reference model
differently from the published export (``benchmark._stated_problem``).

The patch's own guarantees are in ``tests/test_reference_patch.py``; these read the server's
held-out format (``petab_mcp.validation_data``), so they run against the real server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gemot import benchmark as B
from petab_mcp import validation_data as V
from test_reference_patch import PATCH, _toy


def _curated(tmp_path, condition):
    d = tmp_path / "curated"
    d.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "observableId": ["obs_B", "obs_B"],
            "simulationConditionId": [condition] * 2,
            "time": [0.0, 1.0],
            "measurement": [1.0, 1.0],
        }
    ).to_csv(d / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame({"conditionId": [condition]}).to_csv(
        d / V.HELD_OUT_CONDITIONS, sep="\t", index=False
    )
    return "curated/measurements.tsv"


def test_a_transfer_set_is_refit_on_the_stated_problem(tmp_path, monkeypatch):
    """The refit problem is the stated one, so the reference moves the patched initial value the
    way the transfer parameter table asks it to."""
    from gemot import simulate

    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    (tmp_path / "problem").mkdir()
    seen = {}

    def _fake_training(problem, *, path, held_out=None, **_):
        seen["held_out"] = held_out
        ref = {"nll": 1.0, "k": 2, "n": 6}
        Path(path).write_text(json.dumps(ref))
        return ref

    def _fake_refit(problem, *, build_dir, model_name, n_starts, units=None):
        seen["refit_model"] = problem.model
        seen["refit_parameters"] = problem.parameter_df
        return {"status": "ok", "mode": "transfer", "nll": 3.0, "k": 2, "n": 2}

    monkeypatch.setattr(simulate, "write_reference_from_problem", _fake_training)
    monkeypatch.setattr(V, "refit_problem", _fake_refit)
    table = _curated(tmp_path, "c_new")
    (tmp_path / "curated" / V.HELD_OUT_PARAMETERS).write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "V0\tlin\t0.01\t100.0\t2.0\t1\n"
    )

    B.write_reference_for(
        name="X", example_dir=tmp_path, validation_table=table, reference_patch=PATCH
    )

    assert seen["held_out"] is None, "a transfer set is refit, not scored at the training vector"
    assert seen["refit_model"].has_entity_with_id("V0")
    assert seen["refit_model"].sbml_model.getInitialAssignmentBySymbol("k2") is not None
    # the held-out table may ESTIMATE a constant the patch added; the published TABLE has a
    # counterpart for it because `_stated_problem` states one
    assert int(seen["refit_parameters"].loc["V0", "estimate"]) == 1


def test_a_patch_with_a_fixed_theta_set_is_refused(tmp_path, monkeypatch):
    """A fixed-θ set is scored on the pristine structure, so declaring a patch beside one would
    drop it without saying so."""
    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    (tmp_path / "problem").mkdir()
    table = _curated(tmp_path, "c_new")

    with pytest.raises(ValueError, match="only applied where the reference is REFIT"):
        B.write_reference_for(
            name="X", example_dir=tmp_path, validation_table=table, reference_patch=PATCH
        )


def test_materialize_checks_the_held_out_set_against_the_stated_problem(tmp_path, monkeypatch):
    """The other entry point. `materialize` runs the same held-out checks as `write_reference_for`
    and must hand them the same stated problem, or a transfer table that estimates a patch-added
    constant is refused on the setup path while passing on the refit one."""
    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    table = _curated(tmp_path, "c_new")
    (tmp_path / "curated" / V.HELD_OUT_PARAMETERS).write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "V0\tlin\t0.01\t100.0\t2.0\t1\n"
    )

    B.materialize(
        name="X",
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=tmp_path,
        validation_table=table,
        reference_patch=PATCH,
    )

    held = V.read_held_out(B.validation_measurements_path(tmp_path))
    assert held.mode == V.MODE_TRANSFER
    assert int(held.parameter_df.loc["V0", "estimate"]) == 1
