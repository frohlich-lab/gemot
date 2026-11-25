"""``python -m gemot.sweep submit``: an application checkout's examples x models x temperatures x
repeats, submitted to Slurm as one job array whose fits run on the checkout's fit pool.

Each array task runs one ``examples/<example>/run.py`` (``array_task.sh``, which stages the task
into a private copy of its example; the comment at its top says why). This only writes a combos
file and calls ``sbatch``: the work happens in the tasks, never on the login node.

Usage, from the checkout's root:

.. code-block:: bash

    python -m gemot.sweep submit --driver scripted --no-phoenix   # 1-task smoke test, no LLM
    python -m gemot.sweep submit --driver deepagents --repeats 5

Each example contributes its own start count and per-start cap (``AGENT_N_STARTS``,
``AGENT_PER_START_TIMEOUT`` in its ``run.py``), and the sweep runs every example the checkout ships.
Deliberately absent is what only a benchmark sweep needs: a roster, the reference refits' start
counts, re-materializing problems before a sweep, and a reserve for a held-out refit, which would
size a call by the fit budget a pool no longer bounds (gemot#106).

**Nothing here knows the cluster.** A sweep asks Slurm for no partition, excludes no node and
assumes no per-user cap unless told. Slurm's own ``SBATCH_ACCOUNT``, ``SBATCH_PARTITION`` and
``SBATCH_QOS`` in the environment reach every job (the tasks, the service jobs and the pool's
worker jobs, which inherit them), and the flags below override them for the tasks.

**An application's settings are its own input.** Its model, temperature, effort, memory, time, job
name, and its cluster's partition, excluded nodes and per-user cap are flags. They are read first
from its ``slurm/sweep.args`` when it keeps one (one or more flags a line, ``#`` comments), so a
flag on the command line still wins, and an application's sweep never runs gemot's defaults
because a flag was forgotten.

**The fits run on the pool** (``fit_pool_ctl.sh``), so the tasks change with it:

* submission is refused unless the pool's address is published and reachable, as it is unless the
  Phoenix collector's is. Each task reads the address again when it starts, so a task queued
  across a pool restarted on another node joins the new one;
* a task asks only for what its server uses between fits: compiling, scoring at nominal values,
  and the simulations for diagnostics and held-out scores. 8 CPUs, since compiling sets the size
  (``array_task.sh`` caps ``AMICI_PARALLEL_COMPILE`` at 8, and with fewer every compile is slower),
  where it held 100 to run the fits' starts. Its 16G is sized from measured MaxRSS: no archived
  task at 8 CPUs passed 5.6G, and that still held 8 fit processes;
* with ``--qos-cpus``, a per-user CPU cap, the cap is split between the pool's worker jobs and
  the tasks: the throttle is what the pool's ``FITPOOL_MAX_JOBS`` x ``FITPOOL_WORKER_CORES`` (and
  its own job) leave, and a larger one is refused. Tasks holding the cap would keep the worker jobs
  pending, and every fit would wait for a worker while its task's wall time ran out.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from .. import models
from ..ablations import ABLATIONS
from . import SCRIPTS, fit_pool_dir, settings_file, state_dir

#: ``run.py --driver``'s and ``--reasoning-effort``'s choices (``runner/cli.py``; tests/test_sweep.py
#: holds them equal), restated so that this imports without the runner. A typo is refused here
#: rather than by every task at its start.
DRIVERS = ["deepagents", "codex", "claude", "scripted", "barebones"]
REASONING_EFFORTS = ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]


def parse_slurm_time(spec: str) -> int:
    """``HH:MM:SS`` or ``D-HH:MM:SS`` in seconds."""
    days = 0
    if "-" in spec:
        day_part, spec = spec.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"--time {spec!r}: expected HH:MM:SS or D-HH:MM:SS")
    hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def run_constant(example_dir: Path, name: str):
    """A literal numeric constant from an example's ``run.py`` (``AGENT_N_STARTS``,
    ``AGENT_PER_START_TIMEOUT``), read by AST rather than import: importing a run.py pulls the
    server, AMICI and pyPESTO. ``None`` when absent."""
    tree = ast.parse((example_dir / "run.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == name
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
            ):
                return node.value.value
    return None


def examples(repo_root: Path) -> list[str]:
    """Every example the checkout ships: a directory under ``examples/`` with a ``run.py``."""
    return sorted(p.parent.name for p in (Path(repo_root) / "examples").glob("*/run.py"))


def build_combos(
    examples_dir: Path,
    examples,
    models_,
    temperatures,
    reasoning_effort,
    repeats,
    driver,
    n_starts,
    recursion_limit,
    max_wall_seconds,
):
    """Cartesian product -> list of combo dicts. ``models_``/``temperatures`` collapse to a single
    placeholder for the scripted driver, which calls no model. An example's ``AGENT_N_STARTS``
    raises its start count, and its ``AGENT_PER_START_TIMEOUT`` sets its Fides per-start cap."""
    if driver == "scripted":
        models_, temperatures = [models_[0]], [temperatures[0]]
    combos = []
    for example, model, temperature, repeat in itertools.product(
        examples, models_, temperatures, range(repeats)
    ):
        starts = max(n_starts, run_constant(examples_dir / example, "AGENT_N_STARTS") or 0)
        cap = run_constant(examples_dir / example, "AGENT_PER_START_TIMEOUT")
        combos.append(
            dict(
                example=example,
                driver=driver,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                repeat=repeat,
                n_starts=starts,
                recursion_limit=recursion_limit,
                max_wall_seconds=max_wall_seconds,
                per_start_timeout="None" if cap is None else cap,
            )
        )
    return combos


#: The column order ``array_task.sh`` reads.
COMBO_COLUMNS = [
    "idx",
    "example",
    "driver",
    "model",
    "temperature",
    "reasoning_effort",
    "repeat",
    "n_starts",
    "recursion_limit",
    "max_wall_seconds",
    "per_start_timeout",
]


def write_combos_file(combos, path: Path) -> None:
    with path.open("w") as fh:
        fh.write("\t".join(COMBO_COLUMNS) + "\n")
        for i, c in enumerate(combos):
            row = [str(i)] + [str(c[k]) for k in COMBO_COLUMNS[1:]]
            fh.write("\t".join(row) + "\n")


def _arm_exports(specs: list[str]) -> list[str]:
    """``--arm NAME=on|off`` as the ``VAR=value`` strings sbatch's ``--export`` carries, the last
    of one arm winning (so a flag overrides ``sweep.args``); FATAL on anything unrecognised, so a
    typo'd arm cannot silently run the baseline."""
    by_name = {a.name: a for a in ABLATIONS}
    known = ", ".join(f"{a.name} (default {'on' if a.default_on else 'off'})" for a in ABLATIONS)
    exports: dict[str, str] = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        name, value = name.strip(), value.strip().lower()
        if not sep or name not in by_name:
            sys.exit(f"submit: --arm {spec!r}: expected NAME=on|off with NAME one of: {known}")
        if value not in ("on", "off"):
            sys.exit(f"submit: --arm {spec!r}: value must be 'on' or 'off', not {value!r}")
        exports[by_name[name].env] = "1" if value == "on" else "0"
    return [f"{env}={value}" for env, value in exports.items()]


def _resolve_phoenix_endpoint(repo_root: Path) -> str:
    """The live collector's endpoint, or exit(2) with what to do about it.

    Resolved HERE, at submit time, and exported to every task, so nothing in a task's inherited
    environment can misroute it (a 40-task A/B lost its traces to a stale ``PHOENIX_WORKING_DIR``
    on 2026-08-16). Read from the one place ``phoenix_server.sh`` publishes
    it, ``slurm/phoenix_data/``, with no override: importing gemot sets ``PHOENIX_WORKING_DIR`` to
    the checkout's ``.phoenix/``, so an override read from it would always point there. The
    endpoint is REACHED, not just read: endpoint.txt survives a crashed collector.
    """
    endpoint_file = state_dir(repo_root) / "phoenix_data" / "endpoint.txt"
    if not endpoint_file.is_file():
        sys.stderr.write(
            f"[submit] FATAL: --phoenix given but no collector endpoint at {endpoint_file}.\n"
            "  Start it:  python -m gemot.sweep phoenix start\n"
            "  Then check: python -m gemot.sweep phoenix status\n"
            "Refusing to submit an untraced sweep: an untraced run cannot be analysed.\n"
        )
        raise SystemExit(2)
    endpoint = endpoint_file.read_text(encoding="utf-8").strip()
    try:
        import urllib.error
        import urllib.request

        # PROXIES DISABLED: the collector is an intra-cluster host.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            opener.open(endpoint, timeout=10).close()
        except urllib.error.HTTPError:
            pass  # a 4xx is the server ANSWERING
    except Exception as exc:  # noqa: BLE001 - any failure to reach it is the same verdict
        sys.stderr.write(
            f"[submit] FATAL: collector endpoint {endpoint} is not reachable ({exc}).\n"
            "  The endpoint file is stale: the collector died or was preempted.\n"
            "  Restart it:  python -m gemot.sweep phoenix start\n"
            "Refusing to submit an untraced sweep.\n"
        )
        raise SystemExit(2) from None
    print(f"[submit] Phoenix collector live at {endpoint}")
    return endpoint


def probe(address: str, timeout: float = 10.0) -> None:
    """Connect to a pool's scheduler at ``tcp://host:port`` and hang up. Raises ``OSError`` when
    nothing listens there, ``ValueError`` when ``address`` is not one. ``array_task.sh`` runs the
    same probe when a task starts."""
    host, sep, port = address.strip().removeprefix("tcp://").rpartition(":")
    if not sep or not host:
        raise ValueError(f"{address!r} is not tcp://host:port")
    with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout):
        pass


def _read_env(path: Path) -> dict[str, str]:
    """``KEY=value`` lines, as ``fit_pool_server.sh`` writes ``pool.env``."""
    pairs = (line.partition("=") for line in path.read_text(encoding="utf-8").splitlines())
    return {key.strip(): value.strip() for key, sep, value in pairs if sep}


def _resolve_fit_pool(pool_dir: Path) -> dict[str, str]:
    """The live pool's settings, its ``pool.env`` with its ``address``, or exit(2) with what to do.

    The same refusal as the collector's. Without it, every task of the sweep would exit at its
    start (``array_task.sh``'s gate); a pool whose address survived it, after a crash, would hand
    them a dead one.
    """
    address_file = pool_dir / "address.txt"
    if not address_file.is_file():
        sys.stderr.write(
            f"[submit] FATAL: no fit pool address at {address_file}.\n"
            "  Start it:  python -m gemot.sweep fit-pool start\n"
            "  Then check: python -m gemot.sweep fit-pool status\n"
            "Refusing to submit: every task would stop at its start, with nowhere to fit.\n"
        )
        raise SystemExit(2)
    address = address_file.read_text(encoding="utf-8").strip()
    try:
        probe(address)
    except (OSError, ValueError) as exc:
        sys.stderr.write(
            f"[submit] FATAL: the fit pool at {address} is not reachable ({exc}).\n"
            "  The address file is stale: the pool died without removing it.\n"
            "  Restart it:  python -m gemot.sweep fit-pool stop --force\n"
            "               python -m gemot.sweep fit-pool start\n"
            "Refusing to submit.\n"
        )
        raise SystemExit(2) from None
    settings = _read_env(pool_dir / "pool.env") if (pool_dir / "pool.env").is_file() else {}
    missing = [
        k for k in ("FITPOOL_MAX_JOBS", "FITPOOL_WORKER_CORES", "FITPOOL_CPUS") if k not in settings
    ]
    if missing:
        sys.stderr.write(
            f"[submit] FATAL: the fit pool at {address} has no {', '.join(missing)} in "
            f"{pool_dir / 'pool.env'}, which its job writes when it starts. Restart it.\n"
        )
        raise SystemExit(2)
    print(f"[submit] fit pool live at {address}")
    return {**settings, "address": address}


def throttle(args, pool: dict[str, str], n: int) -> int:
    """How many of the sweep's ``n`` tasks run at once: ``--throttle``, or, under a per-user cap
    (``--qos-cpus``), as many as the pool leaves room for, and with neither, all of them."""
    if args.qos_cpus is None:
        print(
            "[submit] no --qos-cpus: nothing splits a per-user CPU cap between these tasks and the "
            "pool's worker jobs. If the account has one, pass it, so that the tasks cannot keep the "
            "workers pending."
        )
        return args.throttle or n
    pool_cpus = int(pool["FITPOOL_MAX_JOBS"]) * int(pool["FITPOOL_WORKER_CORES"]) + int(
        pool["FITPOOL_CPUS"]
    )
    room = (args.qos_cpus - pool_cpus) // args.cpus_per_task
    split = (
        f"the pool's {pool_cpus} CPUs ({pool['FITPOOL_MAX_JOBS']} worker jobs x "
        f"{pool['FITPOOL_WORKER_CORES']}, and {pool['FITPOOL_CPUS']} of its own) leave "
        f"{args.qos_cpus - pool_cpus} of the {args.qos_cpus}-CPU cap"
    )
    if room < 1:
        sys.exit(
            f"submit: {split}: not one task of {args.cpus_per_task} CPUs. Restart the pool with "
            "fewer worker jobs (FITPOOL_MAX_JOBS)."
        )
    if args.throttle is None:
        print(f"[submit] throttle {room}: {split}")
        return room
    if args.throttle > room:
        sys.exit(
            f"submit: --throttle {args.throttle} x {args.cpus_per_task} CPUs exceeds what {split} "
            f"({room} tasks). Tasks holding the cap keep the pool's worker jobs pending, and every "
            "fit waits for a worker while its task's wall time runs out."
        )
    return args.throttle


def build_parser(repo_root: Path, choices: list[str]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gemot.sweep submit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--examples", nargs="+", choices=choices, default=choices)
    p.add_argument(
        "--models",
        nargs="+",
        default=[models.DEFAULT],
        help=f"OpenRouter model id(s) or gemot aliases (default {models.DEFAULT}).",
    )
    p.add_argument("--temperatures", nargs="+", type=float, default=[1.0])
    p.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help="reasoning effort for every model (default: run.py's own, which differs by driver).",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="repeats per (example, model, temperature)."
    )
    p.add_argument(
        "--driver",
        choices=DRIVERS,
        default="scripted",
        help="scripted: no LLM, no network, no API key (default, the cluster smoke test). "
        "deepagents: the gemot panel on OpenRouter (needs OPENROUTER_API_KEY and egress from the "
        "compute nodes). codex, claude: one native CLI session with the server's tools. "
        "barebones: the null arm, one stock agent with a sandboxed shell over the PEtab files.",
    )
    p.add_argument("--n-starts", type=int, default=100)
    # A backstop, not a stopping criterion: gemot's early-stop chain does the real stopping.
    p.add_argument("--recursion-limit", type=int, default=500)

    # --- Slurm resources: the cluster's are the application's (sweep.args), with no default ---
    p.add_argument(
        "--partition",
        default=None,
        help="the tasks' partition (default: SBATCH_PARTITION, else the cluster's default).",
    )
    p.add_argument(
        "--exclude",
        default="",
        help="comma-separated nodes to keep the tasks off, such as nodes whose shared-storage "
        "mount stalls: a task landing on one hangs before it writes a log line.",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="CPUs per task, for its server's compiles (AMICI_PARALLEL_COMPILE is capped at 8) "
        "and simulations; its fits run on the pool.",
    )
    p.add_argument("--mem", default="16G")
    p.add_argument("--time", default="08:00:00")
    p.add_argument(
        "--throttle",
        type=int,
        default=None,
        help="max concurrently running array tasks (default: what the pool leaves of "
        "--qos-cpus, or every task without it).",
    )
    p.add_argument(
        "--qos-cpus",
        type=int,
        default=None,
        help="the account's per-user CPU cap, which the pool's worker jobs and the tasks share "
        "(and the user's other jobs under it, such as the collector).",
    )
    p.add_argument("--job-name", default="gemot-sweep")
    p.add_argument(
        "--wall-time-margin",
        type=int,
        default=1200,
        help="seconds kept BELOW --time for the agent's own wall-time budget "
        "(--max-wall-seconds, unless given), so a run stops and is judged before Slurm kills it.",
    )
    p.add_argument("--max-wall-seconds", type=float, default=None)

    # --- env / logging ---
    p.add_argument(
        "--env-prefix",
        default=sys.prefix,
        help="conda env prefix the tasks activate (default: this interpreter's, which the pool "
        "runs in too).",
    )
    p.add_argument("--wandb", action="store_true", help="log each run to Weights & Biases.")
    p.add_argument("--wandb-project", default="petab-mcp-sweep")
    p.add_argument(
        "--phoenix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export tracing to the ONE shared collector (start it first with "
        "python -m gemot.sweep phoenix start). ON by default; --no-phoenix accepts untraced runs.",
    )
    p.add_argument(
        "--phoenix-project",
        default=None,
        help="group every task under one Phoenix project (default: petab-mcp-<example>).",
    )
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=on|off",
        help="set an ablation arm for every task, e.g. `--arm bio_literature=on`. Repeatable; "
        "`python -m gemot.ablations` lists them.",
    )
    p.add_argument(
        "--out-dir",
        default=str(state_dir(repo_root) / "runs"),
        help="where this sweep's combos file, logs and collected results go.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="write the combos file but do not sbatch."
    )
    return p


def main(argv: list[str], repo_root: Path) -> int:
    repo_root = Path(repo_root).resolve()
    choices = examples(repo_root)
    if not choices:
        sys.exit(f"submit: no examples/<name>/run.py under {repo_root}.")
    settings_path = settings_file(repo_root)
    settings = (
        shlex.split(settings_path.read_text(encoding="utf-8"), comments=True)
        if settings_path.is_file()
        else []
    )
    if settings:
        print(f"[submit] settings from {settings_path}: {shlex.join(settings)}")
    args = build_parser(repo_root, choices).parse_args([*settings, *argv])
    arm_exports = _arm_exports(args.arm)

    time_seconds = parse_slurm_time(args.time)
    if args.max_wall_seconds is not None:
        max_wall_seconds = args.max_wall_seconds
    else:
        max_wall_seconds = time_seconds - args.wall_time_margin
        if max_wall_seconds <= 0:
            print(
                f"[submit] --wall-time-margin ({args.wall_time_margin}s) >= --time "
                f"({time_seconds}s); pass --max-wall-seconds explicitly.",
                file=sys.stderr,
            )
            return 1
    print(
        f"[submit] Slurm --time={args.time} ({time_seconds}s); agent wall-time budget "
        f"{max_wall_seconds:.0f}s"
    )

    combos = build_combos(
        repo_root / "examples",
        args.examples,
        args.models,
        args.temperatures,
        args.reasoning_effort,
        args.repeats,
        args.driver,
        args.n_starts,
        args.recursion_limit,
        max_wall_seconds,
    )
    if not combos:
        print("[submit] empty grid, nothing to submit.", file=sys.stderr)
        return 1

    # Before anything is written, and before --dry-run prints a command a human may paste.
    pool_dir = fit_pool_dir(repo_root)
    pool = _resolve_fit_pool(pool_dir)
    n_parallel = throttle(args, pool, len(combos))
    phoenix_endpoint = _resolve_phoenix_endpoint(repo_root) if args.phoenix else ""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # UNIQUE per submission: array_task.sh reads it at task start, long after sbatch returns, so a
    # later sweep must not overwrite it under queued tasks.
    stamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    combos_path = out_dir / f"combos_{stamp}.tsv"
    write_combos_file(combos, combos_path)
    try:
        shutil.copyfile(combos_path, out_dir / "combos.tsv")  # convenience copy; nothing reads it
    except OSError:
        pass
    print(f"[submit] {len(combos)} combo(s) -> {combos_path}")

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    n = len(combos)
    array_spec = f"0-{n - 1}%{n_parallel}" if n > 1 else "0-0"
    if arm_exports:
        print(f"[submit] ablation arms: {' '.join(arm_exports)}")
    export = ",".join(
        [
            "ALL",
            *([f"PHOENIX_COLLECTOR_ENDPOINT={phoenix_endpoint}"] if phoenix_endpoint else []),
            f"REPO_ROOT={repo_root}",
            f"ENV_PREFIX={args.env_prefix}",
            f"COMBOS_FILE={combos_path}",
            f"RESULTS_DIR={results_dir}",
            f"FITPOOL_DIR={pool_dir}",
            f"WANDB_ENABLED={'1' if args.wandb else '0'}",
            f"WANDB_PROJECT={args.wandb_project}",
            f"PHOENIX_ENABLED={'1' if args.phoenix else '0'}",
            f"PHOENIX_PROJECT={args.phoenix_project or ''}",
            *arm_exports,
        ]
    )
    cmd = [
        "sbatch",
        f"--job-name={args.job_name}",
        *([f"--partition={args.partition}"] if args.partition else []),
        *([f"--exclude={args.exclude}"] if args.exclude.strip() else []),
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        f"--array={array_spec}",
        f"--output={log_dir}/%x_%A_%a.out",
        f"--export={export}",
        str(SCRIPTS / "array_task.sh"),
    ]
    print("[submit] " + " ".join(cmd))
    if args.dry_run:
        print("[submit] --dry-run: not submitting.")
        return 0

    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode
    print(f"[submit] results will be collected under {results_dir}")
    print("[submit] track with: squeue --me")
    return 0
