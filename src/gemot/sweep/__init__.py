"""The Slurm sweep driver: an application checkout's runs as one job array, whose fits share one
pool of SLURM workers.

An application repository pins gemot and holds its own problems as ``examples/<name>/run.py``. It
runs a sweep from its checkout's root, under the environment it built, with ``python -m
gemot.sweep`` (:mod:`.__main__`): ``submit`` (:mod:`.submit`) sends the array, and ``fit-pool``,
``phoenix`` and ``explorer`` start, inspect and stop the three service jobs a sweep uses. The
operator's walk-through is ``docs/SWEEP.md``.

It replaces the copies of a sweep driver that application repositories carried, so there is one
driver. What is an application's own stays in its checkout as input: its examples, its settings
(``slurm/sweep.args``, read before the command line) and the environment it builds (its
``slurm/setup_env.sh``, which cannot come from the package it installs).

Nothing here knows the cluster. Its partition, bad nodes, node size and per-user cap are the
application's settings (``slurm/sweep.args`` for the tasks, ``slurm/fit_pool.env`` for the pool),
or Slurm's own ``SBATCH_ACCOUNT``,
``SBATCH_PARTITION`` and ``SBATCH_QOS``, which every job here inherits. The defaults left are what
the pool and the runs need on any cluster: a task's 8 CPUs, the pool's 2 CPUs and 8 GB, and its
workers' 2 GiB each.

It is a package, not a directory beside ``src/``, because an application installs gemot from git
and has no gemot checkout: the shell scripts beside this module ship in the wheel, and the driver
finds them here. Everything they write goes under the application checkout's ``slurm/``:

* ``slurm/runs/``: each sweep's combos file, task logs and results, and the tasks' work dirs;
* ``slurm/logs/``: the service jobs' logs;
* ``slurm/fit_pool/``: the pool's published address, the settings it runs with (``pool.env``), the
  build root every server of the pool shares (``build/``) and its worker jobs' logs;
* ``slurm/phoenix_data/``, ``slurm/explorer_data/``: the collector's store and endpoint, and the
  explorer's.
"""

from __future__ import annotations

from pathlib import Path

#: The shell scripts: the array task, and each service's job and control script.
SCRIPTS = Path(__file__).resolve().parent


def state_dir(repo_root: Path) -> Path:
    """``<checkout>/slurm``: where every script here writes."""
    return Path(repo_root) / "slurm"


def fit_pool_dir(repo_root: Path) -> Path:
    """``<checkout>/slurm/fit_pool``: the pool's address, settings, build root and worker logs, as
    ``fit_pool_server.sh`` writes them."""
    return state_dir(repo_root) / "fit_pool"


def settings_file(repo_root: Path) -> Path:
    """``<checkout>/slurm/sweep.args``: the application's own ``submit`` flags, when it keeps any."""
    return state_dir(repo_root) / "sweep.args"
