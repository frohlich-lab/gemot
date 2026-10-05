"""Fixtures for the server tier: tests that need the petab-mcp server package itself.

Everything under ``tests/server/`` runs against the REAL server, at the commit ``pyproject.toml``
pins (or a local ``../PEtab-MCP``): the in-process session and engine below, its source and skill
files (``server_src``), or its modules through :mod:`gemot.server_api`, which this tier does NOT
fake. The root ``tests/conftest.py`` leaves this directory out where the server is not
installed; asked for explicitly there, it fails here rather than reporting zero tests as a pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("petab_mcp") is None:
    raise ImportError(
        "tests/server needs the petab-mcp server package: pip install -e '.[server]' "
        "(or -e ../PEtab-MCP); run the server-free tier alone with --ignore=tests/server"
    )


@pytest.fixture(scope="session")
def server_src() -> Path:
    """The installed server package's directory, for tests that read its source or skills. It
    comes from its own repository (frohlich-lab/PEtab-MCP), not ``src/``."""
    return Path(importlib.util.find_spec("petab_mcp").origin).parent


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
def empty_session(empty_yaml, tmp_path):
    """A session whose problem ships model-less — ``has_model()`` is False until ``set_model``."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        empty_yaml,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


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
