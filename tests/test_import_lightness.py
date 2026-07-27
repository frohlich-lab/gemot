"""``petab_harness`` must stay cheap to import.

The harness's top-level ``__init__`` resolves its public names lazily (PEP 562) and every
module the unit tests bind to — the agent's guards, the explorer's bundle reader, the bio
session supervisor, the sweep triage — touches only stdlib at module scope. That is what lets
those tests run with no ``mcp``, ``petab_mcp``, ``phoenix``, ``langchain`` or ``deepagents``
installed, which in turn is why they run at all in a plain ``pip install -e '.[dev]'`` env
instead of silently skipping.

Nothing else notices when that property is lost: adding an eager ``from .runner import ...``
to ``__init__`` or a module-scope ``import phoenix`` to a light submodule keeps every test
passing in a full environment and turns them all into skips in a lean one. This test is the
guard.

Each case runs in a FRESH interpreter: another test in the same session may already have
imported ``mcp``, so ``sys.modules`` in-process proves nothing.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

#: Importing any of these means the module under test stopped being import-light.
_HEAVY = ("mcp", "petab_mcp", "phoenix", "langchain", "langchain_core", "langchain_openai",
          "deepagents", "langgraph", "openai", "wandb", "amici", "pypesto", "petab")

_PROBE = """
import sys
{importline}
heavy = {heavy!r}
leaked = sorted(h for h in heavy if h in {{m.split('.')[0] for m in sys.modules}})
if leaked:
    sys.exit("leaked: " + ", ".join(leaked))
"""


def _assert_light(importline: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(importline=importline, heavy=_HEAVY)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"`{importline}` is no longer import-light:\n{proc.stderr}"


@pytest.mark.parametrize("importline", [
    "import petab_harness",
    "from petab_harness import agent",
    "from petab_harness.agent import dedup, guards",
    "from petab_harness import prompts",
    "from petab_harness import bio_sessions",
    "from petab_harness import experiments",
    "from petab_harness import fit_budget",
    "from petab_harness.explorer import bundle",
    "from petab_harness.runner import endpoints",
])
def test_import_is_light(importline):
    _assert_light(importline)


def _lazy_table(package):
    """A lazy facade's ``{name: relative module}`` map, read from its source without importing
    any submodule (which is the whole point — resolving them needs mcp/petab_mcp)."""
    src = (Path(package.__path__[0]) / "__init__.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_LAZY" for t in node.targets):
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"{package.__name__} has no _LAZY table")


def _resolve(root, target):
    """``'.scenario'`` -> ``scenario.py``; ``'.runner'`` -> ``runner/__init__.py`` (a _LAZY
    target may itself be a package — ``petab_harness._LAZY['run_scenario']`` is one)."""
    stem = target.lstrip(".")
    return (root / stem / "__init__.py") if (root / stem).is_dir() else (root / f"{stem}.py")


def _toplevel_names(path):
    """Every name a module binds at module scope — defs, classes, assignments, imports, plus
    anything its own ``_LAZY`` table resolves if it is itself a facade."""
    tree = ast.parse(Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            if any(getattr(t, "id", None) == "_LAZY" for t in node.targets):
                names |= {k.value for k in node.value.keys}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


@pytest.mark.parametrize("package_name", ["petab_harness", "petab_harness.runner"])
def test_every_lazy_target_actually_defines_its_name(package_name):
    """A ``_LAZY`` entry pointing at the wrong module is invisible until something asks for that
    name — at which point every ``run.py`` dies at import. Checked statically (over the target
    module's AST) so it runs with none of the harness dependencies installed, which is exactly
    the environment where a ``getattr``-based check would have to skip."""
    package = importlib.import_module(package_name)
    root = Path(package.__path__[0])
    for name, target in _lazy_table(package).items():
        module = _resolve(root, target)
        assert module.is_file(), f"{package_name}._LAZY[{name!r}] -> missing {module}"
        assert name in _toplevel_names(module), \
            f"{package_name}._LAZY[{name!r}] -> {module.name} does not define {name!r}"


def test_all_and_lazy_agree():
    """``__all__`` is hand-written; a name added to one and not the other reads as public but
    raises AttributeError (or vice versa: resolvable but undocumented)."""
    import petab_harness
    import petab_harness.runner

    for package in (petab_harness, petab_harness.runner):
        table = _lazy_table(package)
        for name in package.__all__:
            assert name in table or name in vars(package), \
                f"{package.__name__}.__all__ names {name!r}, which nothing defines"


def test_lazy_facade_resolves_by_getattr():
    """The laziness must not cost the ergonomics: ``from petab_harness import Scenario`` still
    works — it just imports ``scenario`` on first access instead of at package import.

    Unlike the static checks above this really drives ``__getattr__``, so it is limited to the
    scenario-backed names; the runner-backed ones need an MCP client to resolve."""
    import petab_harness

    for name, target in _lazy_table(petab_harness).items():
        if target != ".scenario":
            continue
        assert getattr(petab_harness, name) is not None
    assert "Scenario" in dir(petab_harness)
