"""The one door to the petab-mcp server package: every server module gemot reads, by name.

gemot drives the server over MCP, but it also reuses a few of the server's own definitions IN
PROCESS, so a verdict means the same thing on both sides of the wire: the complexity criteria the
judge grades by, the noise-sigma predicates its success gate shares with the server's adequacy
gate, the held-out data format, the skills' version. Those calls come through here and nowhere
else (``tests/test_import_lightness.py`` enforces it), for two reasons:

* ``petab-mcp`` is the ``server`` EXTRA, not a dependency: ``import gemot.<anything>`` must work
  without it, so every accessor imports on first CALL, never at module load.
* The suite outside ``tests/server/`` runs with no server installed. ``tests/conftest.py``
  replaces these accessors with ``tests/fakes/server_api.py``, and
  ``tests/server/test_server_api_fakes.py`` holds each fake to the real module it stands in for.
  A test that reaches past what the fakes provide fails loudly, naming the attribute, which is
  the signal that it belongs in ``tests/server/``.

Each accessor returns the server's MODULE, so a call site reads as the server's own API
(``server_api.criteria().CANONICAL``). Plain constants are not borrowed this way: gemot keeps its
own copy beside the code that uses it, with a drift test in ``tests/server/``
(``runner.scoring``'s noise-sigma thresholds, :mod:`gemot.fit_budget`).
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

#: What to install when a server module is missing — named in the error rather than left to a
#: bare ModuleNotFoundError, which reads like a broken gemot install.
_INSTALL_HINT = (
    "this needs the petab-mcp server package: pip install 'gemot[server]' "
    "(or `pip install -e ../PEtab-MCP` to develop both at once)"
)


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(f"petab_mcp.{name}")
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] == "petab_mcp":
            raise ModuleNotFoundError(f"petab_mcp.{name}: {_INSTALL_HINT}", name=exc.name) from exc
        raise


def require() -> None:
    """Fail NOW, naming the install, if the server package is absent.

    For a caller about to launch the server as a subprocess (:mod:`gemot.runner.main`), where a
    missing package would otherwise surface only as a server that never starts listening, its
    ImportError buried in ``server.log``.
    """
    if importlib.util.find_spec("petab_mcp") is None:
        raise ModuleNotFoundError(f"petab_mcp: {_INSTALL_HINT}", name="petab_mcp")


def criteria() -> ModuleType:
    """:mod:`petab_mcp.criteria`: ``CANONICAL``, ``Criterion``, ``resolve`` — the scoring rules."""
    return _module("criteria")


def fitting() -> ModuleType:
    """:mod:`petab_mcp.fitting`: the noise-sigma predicates and ``needs_flattening``."""
    return _module("fitting")


def identifiability() -> ModuleType:
    """:mod:`petab_mcp.identifiability`: ``effective_parameter_count`` for the reference fit."""
    return _module("identifiability")


def reconcile() -> ModuleType:
    """:mod:`petab_mcp.reconcile`: formula and observable-table parsing."""
    return _module("reconcile")


def skills() -> ModuleType:
    """:mod:`petab_mcp.skills`: the shipped skills' text and version."""
    return _module("skills")


def validation_data() -> ModuleType:
    """:mod:`petab_mcp.validation_data`: the held-out data format and its scoring."""
    return _module("validation_data")
