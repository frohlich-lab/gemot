"""Shared test fixtures and AMICI/SWIG discovery for the test session."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "toy_v1"
YAML_PATH = DATA_DIR / "problem.yaml"


def pytest_configure(config):  # noqa: D401 - pytest hook
    # The server package already owns this (AMICI compiles a C++ model per structure and needs
    # the SWIG the wheel was built against). It used to be copy-pasted here as a third identical
    # implementation; call the shipped one instead.
    from petab_mcp.engine import _ensure_swig_on_path

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
        "model conversion_reaction\n species A = 1;\n k1 = 0.5;\n"
        " r1: A => ; k1 * A;\nend",
        model_id="conversion_reaction",
    )


@pytest.fixture
def session(yaml_path, tmp_path):
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        yaml_path,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
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


@pytest.fixture
def engine(session, amici_build_root):
    from petab_mcp.engine import AmiciEngine

    return AmiciEngine(session, build_root=str(amici_build_root))


@pytest.fixture
def fit_session(session, amici_build_root):
    """A session with the AMICI engine wired and auto-fit on (small n_starts).

    Use for tests that exercise the ``fit`` path; the engine is reachable as
    ``fit_session.engine``.
    """
    from petab_mcp.engine import AmiciEngine

    session.engine = AmiciEngine(session, build_root=str(amici_build_root))
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


