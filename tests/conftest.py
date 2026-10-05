"""Shared test fixtures and AMICI/SWIG discovery for the test session.

The suite has two tiers. Everything directly under ``tests/`` runs WITHOUT the petab-mcp server
package: the server modules gemot reads in process (:mod:`gemot.server_api`) are replaced by
``tests/fakes/server_api.py`` for every test here. ``tests/server/`` holds the tests that need the
real server (its fixtures are in ``tests/server/conftest.py``), and is collected only where it is
installed; the report header says when it was not. CI runs each tier in its own job, so neither
can go quietly missing there.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "toy_v1"
YAML_PATH = DATA_DIR / "problem.yaml"
SERVER_TIER = Path(__file__).parent / "server"
SERVER_INSTALLED = importlib.util.find_spec("petab_mcp") is not None

#: pytest skips a directory named here. Only ever ``server/`` and only without the server: an
#: explicit ``pytest tests/server`` still fails loudly there (see ``tests/server/conftest.py``).
collect_ignore = [] if SERVER_INSTALLED else ["server"]


def pytest_report_header(config):
    if not SERVER_INSTALLED:
        return (
            "tests/server: NOT collected, petab-mcp is not installed "
            "(pip install -e '.[server]', or -e ../PEtab-MCP)"
        )


def pytest_configure(config):  # noqa: D401 - pytest hook
    # AMICI compiles a C++ model per structure and needs the SWIG the wheel was built against.
    # gemot ships this for data regeneration, identical to the server's own copy; calling it
    # rather than the server's keeps this conftest importable without the server.
    from gemot.simulate import _ensure_swig_on_path

    _ensure_swig_on_path()


@pytest.fixture(autouse=True)
def _fake_server_api(request, monkeypatch):
    """Outside ``tests/server/``, every :mod:`gemot.server_api` accessor returns a fake.

    Autouse so no test outside the server tier can reach the real server by forgetting to ask
    for the fake: a test whose path needs more than ``tests/fakes/server_api.py`` provides fails
    with ``NotFaked``, naming the attribute, which says it belongs in ``tests/server/``. A
    subprocess does not inherit this; one that reaches the server is a server-tier test too.
    """
    if SERVER_TIER in request.path.parents:
        return
    from fakes import server_api as fake

    fake.install(monkeypatch.setattr)


@pytest.fixture
def yaml_path() -> str:
    return str(YAML_PATH)


def _langchain_available() -> bool:
    """Is the harness-only ``langchain`` importable, submodule and all?

    Note the two-step probe: when ``langchain`` IS installed, reaching for the submodule may
    raise ModuleNotFoundError from a broken install (a missing transitive dependency), and that
    should surface rather than silently skipping the tests it guards.
    """
    import importlib.util

    if importlib.util.find_spec("langchain") is None:
        return False
    return importlib.util.find_spec("langchain.agents") is not None


#: Skip-marker for tests that build langchain middleware. Lives here because two test modules
#: had byte-identical copies of it.
requires_langchain = pytest.mark.skipif(
    not _langchain_available(), reason="langchain (examples-harness only) not installed"
)


@pytest.fixture
def broken_model():
    """An ``SbmlModel`` missing entities the fixed observable/condition tables need, for tests
    that assert validation rejects it. Built here rather than in the server package: the
    production build path is ``operations.set_model`` -> ``SbmlModel.from_antimony``, so a
    text->model helper is a test fixture, not shipped API."""
    from petab.v1.models.sbml_model import SbmlModel

    return SbmlModel.from_antimony(
        "model conversion_reaction\n species A = 1;\n k1 = 0.5;\n r1: A => ; k1 * A;\nend",
        model_id="conversion_reaction",
    )


@pytest.fixture(scope="session", autouse=True)
def _private_tempdir_per_worker(tmp_path_factory):
    """Give each xdist worker its own ``tempfile.gettempdir()``.

    `amici_build_root` below is already worker-private, but only tests that ASK for it get that
    isolation. Anything constructing a server without an explicit `build_root` --
    `create_server(...)` in test_make_changes/test_server, the commonest shape -- falls back to
    `AmiciEngine`'s default, `tempfile.gettempdir()/petab_mcp_amici`. That is ONE directory shared
    by every worker AND by every concurrent test session on the machine, keyed by the hash of the
    same toy model nearly all of them use. Workers then compile into and clear the same
    `pypesto_<hash>` path, and one of them loses:

        fit rejected: [Errno 2] No such file or directory:
        '/tmp/.../petab_mcp_amici/pypesto_10ba47402a196890'

    reported as an ill-posed model, which is a wrong and expensive diagnosis. Raising `-n` from 4
    to 8 made it show up; it was always a race, just a narrower one. `tempfile.gettempdir()`
    returns `tempfile.tempdir` when set, so pointing that at the worker's own tree fixes every
    caller of the default at once rather than one test helper at a time.
    """
    import tempfile

    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("tmpdir"))
    try:
        yield
    finally:
        tempfile.tempdir = previous


@pytest.fixture(scope="session")
def amici_build_root(tmp_path_factory):
    """One AMICI build root shared by the whole test session.

    The engine caches compiled models by SBML hash, and nearly every test here uses the same
    toy model — but a function-scoped ``tmp_path`` gave each test a private build root, so the
    cache could never hit and every test paid a fresh ~60-90s C++ compile (fit tests twice
    over, since pyPESTO builds its own module). Sharing the root lets a structure compile once
    per session. Nothing leaks between tests: the root holds only compiled artifacts, keyed by
    the hash of the model that produced them, and a genuinely different model still gets its
    own build directory.
    """
    return tmp_path_factory.mktemp("amici_build_root")


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
    for name in (
        "problem.yaml",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
    ):
        shutil.copy(DATA_DIR / name, work / name)
    _write_empty_sbml(work / "conversion_reaction.xml")
    return str(work / "problem.yaml")
