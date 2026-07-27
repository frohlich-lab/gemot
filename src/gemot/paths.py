"""Repo-relative locations the harness reads and writes.

The harness used to live *inside* ``examples/``, so every path it needed (the examples
tree, the run archive, the metrics history, the Phoenix working dir) was a
``Path(__file__).parents[n]`` walk. Now that it is an installed package under ``src/``
those walks would land in the wrong place, so the resolution lives here instead — one
module, one rule, and a single env var to override it.

The harness is a *repo* tool: it runs ``examples/<name>/run.py`` as a subprocess, archives
results into ``examples/_runs_archive/`` and reads an example's shipped ``problem/``. So
"where is the repo" is a real question it has to answer, and the answer is normally the
editable install's own location (``pip install -e .``, which every documented workflow
uses — see the README and ``slurm/setup_env.sh``).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

#: Explicit override, checked first. Set this when the resolution below cannot work:
#: a non-editable (wheel) install, or a run staged into a scratch directory that still
#: has to archive back to the real checkout (``slurm/array_task.sh``).
REPO_ROOT_ENV = "PETAB_MCP_REPO_ROOT"


def _looks_like_repo(path: Path) -> bool:
    """A checkout, not just any directory: it has this project's pyproject AND examples/."""
    return (path / "pyproject.toml").is_file() and (path / "examples").is_dir()


@functools.lru_cache(maxsize=1)
def repo_root() -> Path:
    """The petab-mcp checkout this harness belongs to.

    Resolution order: the ``PETAB_MCP_REPO_ROOT`` override, then the editable install's own
    location (``<root>/src/petab_harness/paths.py`` → ``parents[2]``), then a walk up from
    the working directory, and finally the working directory itself so nothing raises at
    import time — a caller that actually needs the tree will fail on the missing path with
    a clearer error than a resolution exception here would give.
    """
    env = os.environ.get(REPO_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parents[2]
    if _looks_like_repo(here):
        return here

    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
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
    "perelson", "blasi", "rahman", "crauste", "armistead", "sneyd", "fiedler",
    "boehm", "brannmark", "schwen", "bruno", "oliveira", "fujita",
]

#: Every example with a ``run.py``, synthetic demos first (the sweep's rollout order).
ALL_EXAMPLES = [*SYNTHETIC_EXAMPLES, *BENCHMARK_EXAMPLES]


def runs_archive_root() -> Path:
    """``examples/_runs_archive/`` — where every finished run is archived for the explorer."""
    return examples_root() / "_runs_archive"


def metrics_history() -> Path:
    """``examples/metrics_history.csv`` — the cross-run metrics log (gitignored, generated)."""
    return examples_root() / "metrics_history.csv"


def phoenix_working_dir() -> Path:
    """``<repo>/.phoenix`` — kept in the repo rather than ``~/.phoenix`` (see ``__init__``)."""
    return repo_root() / ".phoenix"
