"""``gemot`` must stay cheap to import.

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
_HEAVY = (
    "mcp",
    "petab_mcp",
    "phoenix",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "deepagents",
    "langgraph",
    "openai",
    "wandb",
    "amici",
    "pypesto",
    "petab",
)


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
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"`{importline}` is no longer import-light:\n{proc.stderr}"


@pytest.mark.parametrize(
    "importline",
    [
        "import gemot",
        "from gemot import agent",
        "from gemot.agent import dedup, guards",
        "from gemot.agent import sandbox",
        "from gemot.agent import barebones",
        "from gemot import prompts",
        "from gemot import bio_sessions",
        "from gemot import experiments",
        "from gemot import analysis",
        "from gemot import fit_budget",
        "from gemot.explorer import bundle",
        "from gemot.runner import endpoints",
        "from gemot import server_api",
        "from gemot.runner import scoring, provenance",
        "from gemot import neural_ode",
        "from gemot.neural_ode import route, config, worker",
    ],
)
def test_import_is_light(importline):
    _assert_light(importline)


_ROOT = Path(__file__).resolve().parents[1]


def _server_imports(paths) -> list[str]:
    """``file:line`` of every ``import petab_mcp...`` / ``from petab_mcp... import`` in ``paths``,
    found statically over the AST, so no import is attempted here either."""
    offenders = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            module = (node.module if isinstance(node, ast.ImportFrom) else None) or ""
            names = (
                [a.name for a in getattr(node, "names", [])]
                if isinstance(node, (ast.Import, ast.ImportFrom))
                else []
            )
            if module.split(".")[0] == "petab_mcp" or any(
                n.split(".")[0] == "petab_mcp" for n in names
            ):
                offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    return offenders


def test_only_server_api_imports_the_server_package():
    """Not even LAZILY, inside a function. ``petab_mcp`` is the ``server`` extra: gemot installs,
    imports and runs its server-free tests without it, and every in-process use of it goes through
    :mod:`gemot.server_api`, which is what the server-free tier fakes.

    The parametrized module-scope check above cannot see a `from petab_mcp... import` inside a
    function body, and one of those in the finish gate's ARM step meant that where petab_mcp was
    absent EVERY armed finish raised — and the driver bounces a finish that did not confirm, so the
    run looped to the recursion limit instead of finishing at its honest score. A direct import
    anywhere else in ``src/`` would also bypass the fakes, and only the server-free CI job would
    notice.
    """
    paths = sorted(p for p in (_ROOT / "src" / "gemot").rglob("*.py") if p.name != "server_api.py")
    offenders = _server_imports(paths)
    assert not offenders, f"petab_mcp imported outside gemot.server_api at {offenders}"


def test_only_the_server_tier_imports_the_server_package():
    """A test outside ``tests/server/`` that imports ``petab_mcp`` cannot run where the server is
    not installed, which is the point of that tier; it belongs in ``tests/server/``, or should read
    the server through :mod:`gemot.server_api` and the fakes."""
    paths = sorted(
        p for p in (_ROOT / "tests").rglob("*.py") if (_ROOT / "tests" / "server") not in p.parents
    )
    offenders = _server_imports(paths)
    assert not offenders, f"petab_mcp imported outside tests/server/ at {offenders}"


def _lazy_table(package):
    """A lazy facade's ``{name: relative module}`` map, read from its source without importing
    any submodule (which is the whole point — resolving them needs mcp/petab_mcp)."""
    src = (Path(package.__path__[0]) / "__init__.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_LAZY" for t in node.targets
        ):
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"{package.__name__} has no _LAZY table")


def _resolve(root, target):
    """``'.scenario'`` -> ``scenario.py``; ``'.runner'`` -> ``runner/__init__.py`` (a _LAZY
    target may itself be a package — ``gemot._LAZY['run_scenario']`` is one)."""
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
        # An ANNOTATED module constant (`ABLATIONS: tuple[Ablation, ...] = (...)`) is an AnnAssign,
        # not an Assign, so it needs its own branch — otherwise a script importing it is reported as
        # importing a name that does not exist. A false positive here is expensive: it accuses
        # working code, and the quickest way to silence it is to delete the annotation.
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


@pytest.mark.parametrize("package_name", ["gemot", "gemot.runner"])
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
        assert name in _toplevel_names(module), (
            f"{package_name}._LAZY[{name!r}] -> {module.name} does not define {name!r}"
        )


def test_all_and_lazy_agree():
    """``__all__`` is hand-written; a name added to one and not the other reads as public but
    raises AttributeError (or vice versa: resolvable but undocumented)."""
    import gemot
    import gemot.runner

    for package in (gemot, gemot.runner):
        table = _lazy_table(package)
        for name in package.__all__:
            assert name in table or name in vars(package), (
                f"{package.__name__}.__all__ names {name!r}, which nothing defines"
            )


def test_lazy_facade_resolves_by_getattr():
    """The laziness must not cost the ergonomics: ``from gemot import Scenario`` still
    works — it just imports ``scenario`` on first access instead of at package import.

    Unlike the static checks above this really drives ``__getattr__``, so it is limited to the
    scenario-backed names; the runner-backed ones need an MCP client to resolve."""
    import gemot

    for name, target in _lazy_table(gemot).items():
        if target != ".scenario":
            continue
        assert getattr(gemot, name) is not None
    assert "Scenario" in dir(gemot)


# --------------------------------------------------------------- the fit-budget duplication
def test_fit_workers_honours_an_explicit_override():
    """`n_workers` must beat the import-time FIT_N_WORKERS cap.

    The reference refit for a stiff model (fujita, sneyd) has to run SEQUENTIALLY -- a parallel
    pyPESTO worker hard-crashes with BrokenProcessPool. That used to be requested by setting
    PETAB_MCP_FIT_N_WORKERS at runtime, which silently did nothing: examples/setup.py imports the
    example's run.py (binding fit_budget.FIT_N_WORKERS) BEFORE it could set the env var, so the
    refit kept running in parallel. An argument cannot lose that race; this pins that it is
    actually plumbed through to the engine.
    """
    import os

    from gemot import fit_budget

    assert fit_budget.fit_workers(50, 1) == 1
    assert fit_budget.fit_engine(50, 1) is None  # None == pyPESTO's sequential engine
    # ...and an override never invents parallelism beyond the number of starts.
    assert fit_budget.fit_workers(1, 99) == 1

    # Omitting it preserves the default (env-var-or-core-count) behaviour exactly.
    assert fit_budget.fit_workers(50) == max(1, min(fit_budget.FIT_N_WORKERS, 50))

    # And the env var remains inert once bound, which is the whole reason the argument exists.
    before = fit_budget.FIT_N_WORKERS
    os.environ["PETAB_MCP_FIT_N_WORKERS"] = str(before + 7)
    try:
        assert fit_budget.FIT_N_WORKERS == before
    finally:
        os.environ.pop("PETAB_MCP_FIT_N_WORKERS", None)


def test_simulate_does_not_need_the_server_package():
    """``simulate.py`` may import ``petab``/``amici`` — it is a simulation module — but NOT
    ``petab_mcp``.

    Its docstring promises "only ``amici`` + ``petab`` (not the petab-mcp server package), so data
    can be regenerated from a bare simulation environment", and the ``fit_budget`` duplication above
    is justified by that same promise. Nothing checked it, so a change that replaced simulate's
    self-contained build-env helper with an import of ``petab_mcp.engine`` broke the contract while
    every test stayed green. This pins the one module the ``_HEAVY`` list cannot cover, because
    simulate legitimately imports petab at module scope and so can never be "light" wholesale.
    """
    probe = (
        "import sys\n"
        "from gemot import simulate\n"
        "simulate._ensure_swig_on_path()\n"  # the path that regressed: exercise it, not just import
        "leaked = [m for m in sys.modules if m.split('.')[0] == 'petab_mcp']\n"
        "sys.exit('leaked petab_mcp: ' + ', '.join(sorted(leaked)[:3]) if leaked else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "gemot.simulate must not pull in the server package:\n" + proc.stderr
    )
