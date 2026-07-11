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

    Use for tests that exercise the fit-on-commit path; the engine is reachable as
    ``fit_session.engine``.
    """
    from petab_mcp.engine import AmiciEngine

    session.engine = AmiciEngine(session, build_root=str(tmp_path / "amici"))
    session.auto_fit = True
    session.auto_fit_starts = 2
    return session
