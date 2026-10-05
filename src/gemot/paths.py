"""Locations in the benchmark checkout the harness reads and writes.

The harness is installed from its own repository, but it is a *checkout* tool: it runs
``examples/<name>/run.py`` as a subprocess, archives results into ``examples/_runs_archive/`` and
reads an example's shipped ``problem/``, all in the checkout that pins it (frohlich-lab/gemot_iclr_2026).
So "where is the checkout" is a real question it has to answer, and the resolution lives here —
one module, one rule, and a single env var to override it.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

#: Explicit override, checked first. Set this when neither the running script nor the working
#: directory is inside the checkout: a run staged into a scratch directory that still has to
#: archive back to it (``slurm/array_task.sh``).
REPO_ROOT_ENV = "PETAB_MCP_REPO_ROOT"


def _looks_like_repo(path: Path) -> bool:
    """A checkout, not just any directory: it has a pyproject AND examples/."""
    return (path / "pyproject.toml").is_file() and (path / "examples").is_dir()


@functools.lru_cache(maxsize=1)
def repo_root() -> Path:
    """The benchmark checkout this run belongs to.

    Resolution order: the ``PETAB_MCP_REPO_ROOT`` override, then a walk up from the running
    script (``examples/<name>/run.py``, ``slurm/*.py``), then one from the working directory, and
    finally the working directory itself so nothing raises at import time — a caller that
    actually needs the tree will fail on the missing path with a clearer error than a resolution
    exception here would give.
    """
    env = os.environ.get(REPO_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()

    cwd = Path.cwd().resolve()
    script = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else cwd
    for start in (script, cwd):
        for cand in (start, *start.parents):
            if _looks_like_repo(cand):
                return cand
    return cwd


def examples_root() -> Path:
    """``examples/`` — the scenario tree the harness drives."""
    return repo_root() / "examples"


#: Synthetic (truth-model) demos: they simulate their own data via their own
#: ``generate_data.py`` and are NOT materialized from the pinned benchmark collection.
SYNTHETIC_EXAMPLES = ["enzyme_kinetics", "predator_prey", "repressilator"]

#: Collection-based examples, in rollout order. Each has a ``run.py`` exposing
#: ``BENCHMARK_NAME`` + ``SHIPPED_PARAM_IDS``.
#:
#: THE authoritative list. It used to be copied by hand into ``examples/setup.py``,
#: ``slurm/make_sweep.py`` and ``slurm/refit_references.sh``, which drifted to 13/15/12
#: entries — ``fujita`` fell out of two of them, and since ``make_sweep.py`` passes its copy
#: as argparse ``choices=``, ``--examples fujita`` became a hard error. Add a new example
#: here and nowhere else.
BENCHMARK_EXAMPLES = [
    "perelson",
    "blasi",
    "rahman",
    "crauste",
    "armistead",
    "sneyd",
    "fiedler",
    "boehm",
    "brannmark",
    "schwen",
    "bruno",
    "oliveira",
    "fujita",
    # Large published networks (25-36 species, 36-96 reactions, reference k 46-142). Slower to
    # compile and fit than the rest of the suite -- see LARGE_EXAMPLES below and the wall-time
    # warning in slurm/make_sweep.py.
    "isensee",
    "alkan",
    "bachmann",
    "lucarelli",
    "raimundez",
]

#: The subset needing a bigger per-task budget than the sweep defaults: AMICI has to compile
#: 36-96 reactions before the multistart starts. Derived from the tail of BENCHMARK_EXAMPLES so
#: it cannot drift from the authoritative list above.
LARGE_EXAMPLES = frozenset(BENCHMARK_EXAMPLES[-5:])

#: Model-free data packages: curated PEtab v2 problems imported from frohlich-lab/petab_datasets,
#: with no published mechanistic model behind them. Their ``problem/`` is committed, NOT
#: materialized from the benchmark collection, and they ship no ``reference.json``: the judge grades
#: them reference-free (``Scenario.reference_free``). Kept out of BENCHMARK_EXAMPLES on purpose:
#: everything that walks that list (materialization, the reference refits, LARGE_EXAMPLES' tail)
#: needs a published model to fit.
DATA_PACKAGE_EXAMPLES = ["dessauges", "fontaine", "lun", "thomas"]

#: Every example with a ``run.py``, synthetic demos first, data packages last (the sweep's rollout
#: order).
ALL_EXAMPLES = [*SYNTHETIC_EXAMPLES, *BENCHMARK_EXAMPLES, *DATA_PACKAGE_EXAMPLES]


def runs_archive_root() -> Path:
    """``examples/_runs_archive/`` — where every finished run is archived for the explorer."""
    return examples_root() / "_runs_archive"


def metrics_history() -> Path:
    """``examples/metrics_history.csv`` — the cross-run metrics log (gitignored, generated)."""
    return examples_root() / "metrics_history.csv"


def phoenix_working_dir() -> Path:
    """``<repo>/.phoenix`` — kept in the repo rather than ``~/.phoenix`` (see ``__init__``)."""
    return repo_root() / ".phoenix"
