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
    "from petab_harness.agent import sandbox",
    "from petab_harness.agent import barebones",
    "from petab_harness import prompts",
    "from petab_harness import bio_sessions",
    "from petab_harness import experiments",
    "from petab_harness import analysis",
    "from petab_harness import fit_budget",
    "from petab_harness.explorer import bundle",
    "from petab_harness.runner import endpoints",
    "from petab_harness import neural_ode",
])
def test_import_is_light(importline):
    _assert_light(importline)


def test_the_agent_package_never_imports_the_server_package():
    """Not even LAZILY, inside a function. ``petab_mcp`` is a server-side dependency: the harness
    process that runs the panel talks to the server over MCP and need not have it installed.

    The parametrized module-scope check above cannot see a `from petab_mcp... import` inside a
    function body, and one of those in the finish gate's ARM step meant that where petab_mcp was
    absent EVERY armed finish raised — and the driver bounces a finish that did not confirm, so the
    run looped to the recursion limit instead of finishing at its honest score. Scanned statically
    over the AST, so no import is attempted here either.
    """
    offenders = []
    for path in sorted((Path(__file__).parent.parent / "src" / "petab_harness" / "agent")
                       .rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            module = (node.module if isinstance(node, ast.ImportFrom) else None) or ""
            names = [a.name for a in getattr(node, "names", [])] if isinstance(
                node, (ast.Import, ast.ImportFrom)) else []
            if module.split(".")[0] == "petab_mcp" or any(
                    n.split(".")[0] == "petab_mcp" for n in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"petab_harness.agent imports petab_mcp at {offenders}"


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


# --------------------------------------------------------------- the fit-budget duplication
# `petab_harness.fit_budget` re-implements six things `petab_mcp.fitting` also defines
# (_env_float/_env_int, the default worker count, FIT_PER_START_TIMEOUT_S, FIT_N_WORKERS,
# fides_maxtime_options, fit_engine). That duplication looks removable but is NOT: `_HEAVY`
# above lists `petab_mcp`, so importing ANY petab_mcp module from fit_budget -- including a
# stdlib-only `constants` -- fails test_import_is_light, because petab_mcp/__init__ eagerly
# imports session (and so petab + pandas). fit_budget must stay importable from a bare
# amici+petab environment for simulate.py's offline reference fit.
#
# So the copies stay, and this pins them together instead: the drift the dedup would have
# prevented is what actually matters, and the server + harness computing DIFFERENT fit budgets
# is exactly the bug the two files' cross-referencing comments warn about.
def test_fit_budget_matches_the_server():
    """The harness's fit budget must agree with the server's, or a healthy fit gets killed."""
    from petab_harness import fit_budget

    from petab_mcp import fitting

    assert fit_budget.FIT_PER_START_TIMEOUT_S == fitting.FIT_PER_START_TIMEOUT_S
    assert fit_budget.FIT_N_WORKERS == fitting.FIT_N_WORKERS
    assert fit_budget.fides_maxtime_options() == fitting._fides_maxtime_options()

    # The worker count each side derives for a given multistart, including the edge cases
    # (0/negative starts, fewer starts than workers) where the two expressions differ in shape.
    for n_starts in (0, 1, 2, 3, 7, 50, 1000, -5):
        harness = fit_budget.fit_workers(n_starts)
        server = max(1, min(fitting.FIT_N_WORKERS, int(n_starts)))
        assert harness == server, f"n_starts={n_starts}: harness={harness} server={server}"


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

    from petab_harness import fit_budget

    assert fit_budget.fit_workers(50, 1) == 1
    assert fit_budget.fit_engine(50, 1) is None          # None == pyPESTO's sequential engine
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


# ------------------------------------------------------- out-of-tree importers of our packages
# CI runs pytest, not the scripts under slurm/ and examples/. So when a symbol is deleted from
# src/, a script that imported it stays broken until someone runs it -- which for a slurm job or
# a one-off trace-review probe can be a long time, and (for anything AMICI-backed) only after
# paying for a model compile. This has now happened twice on one branch: `fitting._run_fit(seed=)`
# and `from petab_mcp.model import SbmlModel`.
#
# Resolve every such import statically, against the target module's SOURCE -- no importing, so
# this stays cheap and needs none of the heavy deps those scripts want.
#
# LIMIT, stated plainly: this catches the IMPORT class only. A call site passing a keyword
# argument that no longer exists (the `seed=` break) still slips through -- checking that would
# need real signature binding, which needs the imports this test exists to avoid.
def _repo_scripts():
    root = Path(__file__).resolve().parents[1]
    for pattern in ("slurm/**/*.py", "examples/**/*.py"):
        for path in root.glob(pattern):
            # skip AMICI's generated SWIG wrappers under <example>/.amici_build/
            if ".amici_build" not in path.parts and "_runs_archive" not in path.parts:
                yield root, path


def test_scripts_only_import_names_our_packages_actually_define():
    """`from petab_mcp.X import Y` / `from petab_harness.X import Y` in slurm/ and examples/ must
    name something module X really binds."""
    broken = []
    for root, path in _repo_scripts():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.level or node.module.split(".")[0] not in ("petab_mcp", "petab_harness"):
                continue
            target = root / "src" / Path(node.module.replace(".", "/"))
            source = (target.with_suffix(".py") if target.with_suffix(".py").is_file()
                      else target / "__init__.py")
            if not source.is_file():
                broken.append(f"{path.relative_to(root)}:{node.lineno} -> no module {node.module}")
                continue
            defined = _toplevel_names(source)
            for alias in node.names:
                # `from pkg import submodule` is legal without pkg/__init__ binding the name
                if alias.name == "*" or (source.name == "__init__.py"
                                         and (source.parent / f"{alias.name}.py").is_file()):
                    continue
                if alias.name not in defined:
                    broken.append(
                        f"{path.relative_to(root)}:{node.lineno} -> "
                        f"{node.module} does not define {alias.name!r}")
    assert not broken, "scripts import names that no longer exist:\n  " + "\n  ".join(broken)


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
        "from petab_harness import simulate\n"
        "simulate._ensure_swig_on_path()\n"   # the path that regressed: exercise it, not just import
        "leaked = [m for m in sys.modules if m.split('.')[0] == 'petab_mcp']\n"
        "sys.exit('leaked petab_mcp: ' + ', '.join(sorted(leaked)[:3]) if leaked else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "petab_harness.simulate must not pull in the server package:\n" + proc.stderr)
