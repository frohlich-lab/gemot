"""``python -m gemot.sweep COMMAND``, from the root of an application checkout.

* ``submit [FLAGS]``: build a sweep and submit it as one job array (:mod:`.submit`).
* ``fit-pool {start,status,stop} [--force]``: the fit pool's service job (``fit_pool_ctl.sh``).
* ``phoenix {start,status,stop} [--force]``: the shared trace collector (``phoenix_ctl.sh``).
* ``explorer {start,status,stop}``: the run explorer (``explorer_ctl.sh``).

A service command runs its control script with two things it cannot find for itself, now that it
lives in the installed package rather than in the checkout: the checkout (``REPO_ROOT``, resolved
by :func:`gemot.paths.repo_root`) and the environment (``ENV_PREFIX``, this interpreter's own
``sys.prefix``). So a service job always runs the gemot, and the server, that submitted it: a pool
started from another environment would fail every fit at the server's version check.
"""

from __future__ import annotations

import os
import sys

from .. import paths
from . import SCRIPTS

#: Each service's control script, by command.
SERVICES = {
    "fit-pool": "fit_pool_ctl.sh",
    "phoenix": "phoenix_ctl.sh",
    "explorer": "explorer_ctl.sh",
}


def _checkout():
    """The application checkout, or exit(2): every script here writes under its ``slurm/`` and
    runs its ``examples/``, so a command run from anywhere else would scatter state."""
    root = paths.repo_root()
    if not paths._looks_like_repo(root):
        sys.exit(
            f"gemot.sweep: {root} is not an application checkout (no pyproject.toml and "
            f"examples/). Run from the checkout's root, or set {paths.REPO_ROOT_ENV}."
        )
    return root


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "submit":
        from .submit import main as submit

        return submit(rest, repo_root=_checkout())
    if command not in SERVICES:
        print(f"gemot.sweep: unknown command {command!r}\n{__doc__}", file=sys.stderr)
        return 2
    env = {
        **os.environ,
        "REPO_ROOT": str(_checkout()),
        "ENV_PREFIX": sys.prefix,
    }
    os.execvpe("bash", ["bash", str(SCRIPTS / SERVICES[command]), *rest], env)
    return 1  # not reached: exec replaced this process


if __name__ == "__main__":
    raise SystemExit(main())
