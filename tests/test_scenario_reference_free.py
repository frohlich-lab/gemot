"""The ``Scenario`` half of the reference-free judge: the two fields and what ``reference()`` does.

Import-light on purpose (``gemot.scenario`` needs no ``petab_mcp``), so the contract a data-package
``run.py`` is written against is checked even where the server is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemot import scenario as S


async def _no_fix(call, n_starts):  # pragma: no cover - never driven here
    return None


def _scenario(problem_dir: Path, **kw) -> S.Scenario:
    return S.Scenario(name="pkg", problem_dir=problem_dir, scripted_fix=_no_fix, **kw)


@pytest.mark.parametrize("test", S.ADEQUACY_TESTS)
def test_a_reference_free_scenario_declares_one_of_the_three_tests(tmp_path, test):
    sc = _scenario(tmp_path, reference_free=True, adequacy=test)
    assert sc.reference_free and sc.adequacy == test


@pytest.mark.parametrize("adequacy", [None, "", "bic", "Chi2"])
def test_a_reference_free_scenario_without_a_valid_test_is_refused(tmp_path, adequacy):
    """Without a declared test every run would grade unassessable — a benchmark arm that can never
    pass, discovered only after a sweep. Refused at construction instead."""
    with pytest.raises(ValueError, match="must declare `adequacy`"):
        _scenario(tmp_path, reference_free=True, adequacy=adequacy)


@pytest.mark.parametrize("test", S.ADEQUACY_TESTS)
def test_a_test_on_a_reference_based_scenario_is_refused(tmp_path, test):
    """The judge reads `adequacy` only in reference-free mode, so on any other scenario the field
    would be silently ignored — an author who set it believing it gated the run would be wrong."""
    with pytest.raises(ValueError, match="reference-free scenario"):
        _scenario(tmp_path, adequacy=test)


def test_reference_free_never_reads_reference_json(tmp_path, monkeypatch):
    """There is no true model, so ``reference()`` is None without looking: a stray reference.json in
    a data package's problem/ must not quietly put a bar back under its runs."""
    (tmp_path / "reference.json").write_text(json.dumps({"nll": 1.0, "k": 2, "n": 10}))
    sc = _scenario(tmp_path, reference_free=True, adequacy=S.ADEQUACY_CHI2)

    def _touched(*args, **kwargs):
        raise AssertionError("reference() touched the disk for a reference-free scenario")

    monkeypatch.setattr(Path, "is_file", _touched)
    monkeypatch.setattr(Path, "read_text", _touched)
    assert sc.reference() is None


def test_a_reference_based_scenario_still_reads_its_reference(tmp_path):
    ref = {"nll": 1.0, "k": 2, "n": 10}
    (tmp_path / "reference.json").write_text(json.dumps(ref))
    assert _scenario(tmp_path).reference() == ref


def test_a_missing_reference_does_not_make_a_scenario_reference_free(tmp_path):
    """The safety property: reference-free is DECLARED, never inferred. Deleting an example's answer
    key must keep failing it (the judge's "could not evaluate"), not turn it into an example graded
    on absolute adequacy."""
    sc = _scenario(tmp_path)
    assert sc.reference() is None
    assert sc.reference_free is False and sc.adequacy is None
