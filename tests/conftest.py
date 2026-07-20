"""Shared test fixtures and AMICI/SWIG discovery for the test session."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "toy_v1"
YAML_PATH = DATA_DIR / "problem.yaml"


def _ensure_swig_on_path() -> None:
    if shutil.which("swig"):
        return
    try:
        import swig

        cand = Path(swig.__file__).resolve().parent / "data" / "bin"
        if (cand / "swig").exists():
            os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def pytest_configure(config):  # noqa: D401 - pytest hook
    _ensure_swig_on_path()


@pytest.fixture
def yaml_path() -> str:
    return str(YAML_PATH)


# The shipped toy model (tests/data/toy_v1/model.ant). Structural edits in tests are made by
# re-authoring this base (via ``set_model``) with extra reaction lines spliced in.
_TOY_BASE_MODEL = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
{extra}end
"""


@pytest.fixture
def reauthor_toy():
    """Return a helper that re-authors the toy model with ``extra_reactions`` (Antimony lines)
    spliced in via ``set_model`` and then ``fit``s it, for tests that just need a committed,
    versioned structural change.

    Since issue #81 an edit is fit-free (it snapshots no version); ``fit`` is what commits the
    version. This helper does both so a caller gets a committed version in one call. Engine-free
    (the plain ``session`` fixture wires no engine), ``fit`` still snapshots a ``kind="fit"``
    version recording the edited model — the fit itself is skipped. Returns the ``fit`` result."""
    from petab_mcp import operations as O

    def _reauthor(session, extra_reactions=(), *, justification="test", **kwargs):
        body = "".join(f"  {line}\n" for line in extra_reactions)
        antimony = _TOY_BASE_MODEL.format(extra=body)
        O.set_model(session, antimony, justification=justification, **kwargs)
        return session.fit()

    return _reauthor


@pytest.fixture
def session(yaml_path, tmp_path):
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        yaml_path,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


@pytest.fixture
def engine(session, tmp_path):
    from petab_mcp.engine import AmiciEngine

    return AmiciEngine(session, build_root=str(tmp_path / "amici"))


@pytest.fixture
def fit_session(session, tmp_path):
    """A session with the AMICI engine wired and auto-fit on (small n_starts).

    Use for tests that exercise the ``fit`` path; the engine is reachable as
    ``fit_session.engine``.
    """
    from petab_mcp.engine import AmiciEngine

    session.engine = AmiciEngine(session, build_root=str(tmp_path / "amici"))
    session.auto_fit = True
    session.auto_fit_starts = 2
    return session


def _write_empty_sbml(path: Path, model_id: str = "conversion_reaction") -> None:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(model_id)
    model.setMetaId(model_id)
    libsbml.writeSBMLToFile(doc, str(path))


@pytest.fixture
def empty_yaml(tmp_path) -> str:
    """A PEtab problem that ships WITHOUT a model (empty SBML, no model.ant)."""
    work = tmp_path / "empty"
    work.mkdir()
    for name in ("problem.yaml", "conditions.tsv", "measurements.tsv",
                 "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    _write_empty_sbml(work / "conversion_reaction.xml")
    return str(work / "problem.yaml")


@pytest.fixture
def empty_session(empty_yaml, tmp_path):
    """A session whose problem ships model-less — ``has_model()`` is False until ``set_model``."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        empty_yaml,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
