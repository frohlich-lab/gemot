"""The anonymous arm leaves the numerical objective unchanged: the same likelihood, training and
held out, fit through the real server's session and engine. Hiding the identities is tested in
``tests/test_biology_blinding.py``.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from gemot.biology_blinding import (
    prepare,
    private_dir,
)
from test_biology_blinding import TOY


def test_anonymous_sbml_preserves_training_and_held_out_likelihood(tmp_path, amici_build_root):
    from petab_mcp.engine import AmiciEngine
    from petab_mcp.fitting import fit_summary, score_at_nominal
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession
    from petab_mcp.validation_data import read_held_out, score_session

    validation = tmp_path / "validation"
    validation.mkdir()
    measurements = pd.read_csv(TOY / "measurements.tsv", sep="\t").iloc[:1].copy()
    measurements["time"] = 1.5
    measurements["measurement"] = 0.45
    measurements.to_csv(validation / "measurements.tsv", sep="\t", index=False)
    work = tmp_path / "anonymous"
    prepare(TOY / "problem.yaml", work, validation)
    original = tmp_path / "original_problem"
    shutil.copytree(TOY, original)
    values = []
    heldout_values = []
    for label, yaml in (
        ("original", original / "problem.yaml"),
        ("anonymous", work / "problem.yaml"),
    ):
        session = PetabSession(
            yaml, provenance_dir=tmp_path / label / "provenance", resolver=StubReferenceResolver()
        )
        engine = AmiciEngine(session, build_root=str(amici_build_root))
        score_at_nominal(session, engine)
        values.append(fit_summary(session, engine))
        heldout = validation if label == "original" else private_dir(work) / "validation"
        heldout_values.append(
            score_session(session, engine, read_held_out(heldout / "measurements.tsv"))
        )
    assert values[0]["neg_log_likelihood"] == pytest.approx(
        values[1]["neg_log_likelihood"], abs=1e-9
    )
    assert values[0]["bic"] == pytest.approx(values[1]["bic"], abs=1e-9)
    assert heldout_values[0]["status"] == heldout_values[1]["status"] == "ok"
    assert heldout_values[0]["nll"] == pytest.approx(heldout_values[1]["nll"], abs=1e-9)
