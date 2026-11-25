"""The sweep driver (gemot#107): submit's refusals and what it submits, the fit pool's control
script and service job, and the array task's pool gate.

Nothing here reaches Slurm. The control scripts run against stub ``squeue``/``sbatch``/``scancel``
that keep a jobs table in a file and log every call, and the service job and the array task run
with a stub ``conda`` and stub programs that record what they were handed. A pool that is "live"
is a socket listening on localhost, which is all the probes check.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import socket
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

import gemot
from gemot.sweep import SCRIPTS
from gemot.sweep import submit as S

SRC = Path(gemot.__file__).resolve().parents[1]

#: Nodes a cluster's settings keep jobs off (submit --exclude, FITPOOL_EXCLUDE).
_EXCLUDE = "node017,node042"

_RUN_PY = """\
import json, os, sys
from pathlib import Path
AGENT_N_STARTS = {n_starts}
AGENT_PER_START_TIMEOUT = 600
work = Path(__file__).resolve().parent / "work"
work.mkdir(exist_ok=True)
env = {{k: v for k, v in os.environ.items() if k.startswith(("PETAB_MCP_", "AMICI_"))}}
(work / "result.json").write_text(json.dumps({{"archive_dir": "x", "argv": sys.argv[1:], "env": env}}))
"""

#: What a pool's job writes, as fit_pool_server.sh does with the ctl's defaults.
_POOL_ENV = {"FITPOOL_MAX_JOBS": "12", "FITPOOL_WORKER_CORES": "100", "FITPOOL_CPUS": "2"}


def _checkout(tmp_path: Path, n_starts: int = 150, name: str = "app") -> Path:
    repo = tmp_path / name
    (repo / "examples" / "toy").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    (repo / "examples" / "toy" / "run.py").write_text(_RUN_PY.format(n_starts=n_starts))
    return repo


@contextlib.contextmanager
def _live_pool(repo: Path, settings: dict[str, str] = _POOL_ENV):
    """A listening socket published as the checkout's pool, as fit_pool_server.sh publishes one."""
    pool = repo / "slurm" / "fit_pool"
    pool.mkdir(parents=True, exist_ok=True)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        address = f"tcp://127.0.0.1:{server.getsockname()[1]}"
        (pool / "address.txt").write_text(address + "\n")
        (pool / "pool.env").write_text("".join(f"{k}={v}\n" for k, v in settings.items()))
        yield address


def _dead_address() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


def _submit(repo: Path, *argv: str) -> None:
    """``submit --dry-run --no-phoenix ARGV``, which prints the sbatch command it would run."""
    assert S.main(["--dry-run", "--no-phoenix", *argv], repo_root=repo) == 0


def _sbatch_line(out: str) -> list[str]:
    line = next(line for line in out.splitlines() if line.startswith("[submit] sbatch "))
    return shlex.split(line.removeprefix("[submit] "))


# --------------------------------------------------------------------------- submit


def test_submit_refuses_without_a_published_pool(tmp_path, capsys):
    # Every task would exit at its start with nowhere to fit; refused at submit, as for Phoenix.
    repo = _checkout(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _submit(repo)
    assert exc.value.code == 2
    assert "fit-pool start" in capsys.readouterr().err
    assert not (repo / "slurm" / "runs").exists(), "a refused sweep must leave no state on disk"


def test_submit_refuses_a_pool_whose_address_survived_it(tmp_path, capsys):
    # address.txt outlives a crashed pool: read is not enough, the address must answer.
    repo = _checkout(tmp_path)
    pool = repo / "slurm" / "fit_pool"
    pool.mkdir(parents=True)
    (pool / "address.txt").write_text(_dead_address())
    (pool / "pool.env").write_text("".join(f"{k}={v}\n" for k, v in _POOL_ENV.items()))
    with pytest.raises(SystemExit) as exc:
        _submit(repo)
    assert exc.value.code == 2
    assert "not reachable" in capsys.readouterr().err


def test_submit_tasks_ask_for_8_cpus_and_the_pools_share_of_a_cap(tmp_path, capsys):
    # The 100 CPUs a task held for its fits go: 8, for its compiles. Under a per-user cap of 3000,
    # the throttle is what the pool's 12 x 100 workers and its own 2 CPUs leave.
    repo = _checkout(tmp_path)
    with _live_pool(repo):
        _submit(repo, "--repeats", "3", "--qos-cpus", "3000")
    out = capsys.readouterr().out
    cmd = _sbatch_line(out)
    assert "--cpus-per-task=8" in cmd and "--mem=16G" in cmd
    assert f"--array=0-2%{(3000 - 1202) // 8}" in cmd
    assert cmd[-1] == str(SCRIPTS / "array_task.sh")
    export = dict(kv.split("=", 1) for kv in cmd[-2].removeprefix("--export=").split(",")[1:])
    assert export["FITPOOL_DIR"] == str(repo / "slurm" / "fit_pool")
    assert export["REPO_ROOT"] == str(repo)
    combos = sorted((repo / "slurm" / "runs").glob("combos_*.tsv"))[0].read_text().splitlines()
    assert combos[0].split("\t") == S.COMBO_COLUMNS
    row = dict(zip(S.COMBO_COLUMNS, combos[1].split("\t")))
    # The example's own AGENT_N_STARTS raises --n-starts, and its cap rides along.
    assert (row["n_starts"], row["per_start_timeout"], row["reasoning_effort"]) == (
        "150",
        "600",
        "None",
    )


def test_submit_knows_no_cluster_unless_told(tmp_path, capsys):
    # No partition (Slurm's own, or SBATCH_PARTITION), no excluded node, no per-user cap: one
    # cluster's would send the tasks nowhere, or hold them back, on any other.
    repo = _checkout(tmp_path)
    with _live_pool(repo):
        _submit(repo, "--repeats", "3")
    out = capsys.readouterr().out
    cmd = _sbatch_line(out)
    assert not any(a.startswith(("--partition", "--exclude")) for a in cmd)
    assert "--array=0-2%3" in cmd
    assert "no --qos-cpus" in out


def test_submit_refuses_a_throttle_that_would_starve_the_pool(tmp_path):
    # Tasks holding a per-user cap keep the pool's worker jobs pending: every fit would wait while
    # its task's wall time ran out.
    repo = _checkout(tmp_path)
    with _live_pool(repo), pytest.raises(SystemExit) as exc:
        _submit(repo, "--throttle", "300", "--qos-cpus", "3000")
    assert "keep the pool's worker jobs pending" in str(exc.value.code)


def test_submit_reads_the_applications_settings_first(tmp_path, capsys):
    # An application's own settings live in its slurm/sweep.args, and a flag given on the
    # command line still wins; an arm set twice exports its last value only.
    repo = _checkout(tmp_path)
    (repo / "slurm").mkdir()
    (repo / "slurm" / "sweep.args").write_text(
        "# the application's run configuration\n"
        "--models z-ai/glm-5.3-flash --temperatures 1.0\n"
        "--job-name app-gemot --mem 64G\n"
        f"--partition compute --exclude {_EXCLUDE} --qos-cpus 3000\n"
        "--arm biocurator=off\n"
    )
    with _live_pool(repo):
        _submit(repo, "--mem", "24G", "--arm", "biocurator=on", "--driver", "deepagents")
    out = capsys.readouterr().out
    cmd = _sbatch_line(out)
    assert "--job-name=app-gemot" in cmd and "--mem=24G" in cmd
    assert "--partition=compute" in cmd and f"--exclude={_EXCLUDE}" in cmd
    exports = cmd[-2].split(",")
    arm = next(a for a in S.ABLATIONS if a.name == "biocurator").env
    assert [e for e in exports if e.startswith(f"{arm}=")] == [f"{arm}=1"]
    combos = sorted((repo / "slurm" / "runs").glob("combos_*.tsv"))[0].read_text().splitlines()
    assert dict(zip(S.COMBO_COLUMNS, combos[1].split("\t")))["model"] == "z-ai/glm-5.3-flash"


def test_submit_offers_the_drivers_and_efforts_run_py_accepts():
    # DRIVERS and REASONING_EFFORTS restate runner/cli.py's choices so that a typo fails at
    # submit, not in every task; they must not drift apart, as DRIVERS did when codex and claude
    # joined the runner.
    from gemot.runner.cli import build_parser

    parser = build_parser(types.SimpleNamespace(name="toy"))
    choices = {a.dest: list(a.choices) for a in parser._actions if a.choices}
    assert choices["driver"] == S.DRIVERS
    assert choices["reasoning_effort"] == S.REASONING_EFFORTS


def test_probe_reads_tcp_addresses():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        S.probe(f"tcp://127.0.0.1:{server.getsockname()[1]}\n")
    with pytest.raises(OSError):
        S.probe(_dead_address())
    with pytest.raises(ValueError):
        S.probe("not-an-address")


# --------------------------------------------------------------------------- stub Slurm

_SQUEUE = r"""#!/usr/bin/env bash
# squeue --me [--noheader] --name=A[,B] [--format=%A]: ids (or a table) of the jobs named so.
names=""; ids_only=0
for a in "$@"; do
    case "$a" in --name=*) names="${a#--name=}" ;; --format=%A) ids_only=1 ;; esac
done
echo "squeue $*" >> "$STUB/calls"
[[ $ids_only == 1 ]] || echo "JOBID NAME"
while read -r id name; do
    [[ -z "$id" ]] && continue
    if [[ ",$names," == *",$name,"* ]]; then
        if [[ $ids_only == 1 ]]; then echo "$id"; else echo "$id $name"; fi
    fi
done < "$STUB/jobs"
"""

_SBATCH = r"""#!/usr/bin/env bash
echo "sbatch $*" >> "$STUB/calls"
env > "$STUB/sbatch.env"
name=""
for a in "$@"; do case "$a" in --job-name=*) name="${a#--job-name=}" ;; esac; done
echo "4242 $name" >> "$STUB/jobs"
echo 4242
"""

# A TERM to the pool's batch step closes the cluster, which cancels its worker jobs, unless the test
# says the pool leaves them (STUB/leaves_workers) or ignores the signal (STUB/hangs).
_SCANCEL = r"""#!/usr/bin/env bash
echo "scancel $*" >> "$STUB/calls"
drop() { grep -v -E "^($1) " "$STUB/jobs" > "$STUB/jobs.new" || true; mv "$STUB/jobs.new" "$STUB/jobs"; }
if [[ "$1" == "--signal=TERM" ]]; then
    [[ -f "$STUB/hangs" ]] && exit 0
    drop "$3"
    [[ -f "$STUB/leaves_workers" ]] || { grep -v -- "-worker$" "$STUB/jobs" > "$STUB/jobs.new" || true; mv "$STUB/jobs.new" "$STUB/jobs"; }
    exit 0
fi
drop "$(echo "$*" | tr ' ' '|')"
"""

_CONDA = r"""#!/usr/bin/env bash
# `conda info --base`: a base whose conda.sh defines a `conda` that activates nothing.
echo "$STUB/conda_base"
"""


def _stubs(tmp_path: Path, jobs: str = "") -> tuple[Path, dict[str, str]]:
    """A stub bin dir first on PATH, and the env to run a script with."""
    stub = tmp_path / "stub"
    (stub / "bin").mkdir(parents=True)
    (stub / "conda_base" / "etc" / "profile.d").mkdir(parents=True)
    (stub / "conda_base" / "etc" / "profile.d" / "conda.sh").write_text("conda() { :; }\n")
    for name, body in (
        ("squeue", _SQUEUE),
        ("sbatch", _SBATCH),
        ("scancel", _SCANCEL),
        ("conda", _CONDA),
    ):
        (stub / "bin" / name).write_text(body)
        (stub / "bin" / name).chmod(0o755)
    # A wrapper, not a symlink: a venv's interpreter finds its pyvenv.cfg beside the path it runs as.
    (stub / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (stub / "bin" / "python").chmod(0o755)
    (stub / "jobs").write_text(jobs)
    (stub / "calls").write_text("")
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("FITPOOL_", "PETAB_MCP_", "SBATCH_"))
        },
        "PATH": f"{stub / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(SRC),
        "STUB": str(stub),
    }
    return stub, env


def _ctl(repo: Path, env: dict[str, str], *argv: str, **extra: str):
    return subprocess.run(
        ["bash", str(SCRIPTS / "fit_pool_ctl.sh"), *argv],
        env={**env, "REPO_ROOT": str(repo), "ENV_PREFIX": sys.prefix, **extra},
        capture_output=True,
        text=True,
    )


def _calls(stub: Path, program: str) -> list[str]:
    return [c for c in (stub / "calls").read_text().splitlines() if c.startswith(program)]


#: The pool's cluster-dependent settings, which have no default.
_SIZED = {"FITPOOL_MAX_JOBS": "12", "FITPOOL_WORKER_CORES": "100", "FITPOOL_TIME": "3-00:00:00"}


def _pool_settings(repo: Path, settings: dict[str, str] = _SIZED, extra: str = "") -> None:
    """The application's slurm/fit_pool.env."""
    (repo / "slurm").mkdir(exist_ok=True)
    (repo / "slurm" / "fit_pool.env").write_text(
        "# this cluster's pool\n" + "".join(f"{k}={v}\n" for k, v in settings.items()) + extra
    )


# --------------------------------------------------------------------------- the pool's ctl


def test_pool_start_submits_the_service_job_the_application_sized(tmp_path):
    # The cluster's share comes from the application's slurm/fit_pool.env, and a variable given on
    # the command line wins over it. The pool's own job at 2 CPUs and 8 GB, its workers at 2 GiB
    # each, their walltime the pool's; no partition or node of any cluster's unless set.
    repo = _checkout(tmp_path)
    _pool_settings(repo, extra=f"FITPOOL_EXCLUDE={_EXCLUDE}  # stalling mounts\n")
    stub, env = _stubs(tmp_path)
    run = _ctl(repo, env, "start", FITPOOL_MAX_JOBS="2")
    assert run.returncode == 0, run.stderr
    (sbatch,) = _calls(stub, "sbatch")
    args = shlex.split(sbatch)
    for expected in (
        "--job-name=fitpool-app",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=3-00:00:00",
        f"--exclude={_EXCLUDE}",
        "--export=ALL",
        str(SCRIPTS / "fit_pool_server.sh"),
    ):
        assert expected in args
    assert not any(a.startswith(("--nodelist", "--partition")) for a in args)
    sent = dict(
        line.split("=", 1) for line in (stub / "sbatch.env").read_text().splitlines() if "=" in line
    )
    keys = ("FITPOOL_MAX_JOBS", "FITPOOL_WORKER_CORES", "FITPOOL_WORKER_MEM", "FITPOOL_WORKER_TIME")
    assert {k: sent[k] for k in keys} == {
        "FITPOOL_MAX_JOBS": "2",
        "FITPOOL_WORKER_CORES": "100",
        "FITPOOL_WORKER_MEM": "200GiB",
        "FITPOOL_WORKER_TIME": "3-00:00:00",
    }
    assert sent["FITPOOL_WORKER_JOB"] == "fitpool-app-worker"
    assert (sent["REPO_ROOT"], sent["ENV_PREFIX"]) == (str(repo), sys.prefix)


def test_pool_start_pins_the_partition_and_the_old_node(tmp_path):
    # A pool restarted on its old node comes back at the same address, which running servers keep.
    repo = _checkout(tmp_path)
    _pool_settings(repo)
    stub, env = _stubs(tmp_path)
    run = _ctl(repo, env, "start", FITPOOL_NODELIST="node007", FITPOOL_PARTITION="compute")
    assert run.returncode == 0, run.stderr
    args = shlex.split(_calls(stub, "sbatch")[0])
    assert "--nodelist=node007" in args and "--partition=compute" in args


@pytest.mark.parametrize(
    ("settings", "why"),
    [
        ({"FITPOOL_TIME": ""}, "set FITPOOL_TIME for this cluster"),
        ({"FITPOOL_WORKER_TIME": "1-00:00:00"}, "never replaced"),
        ({"FITPOOL_MIN_JOBS": "13"}, "FITPOOL_MIN_JOBS"),
        ({"FITPOOL_MAX_JOBS": "twelve"}, "not a whole number"),
        ({"FITPOOL_TIME": "3d"}, "not D-HH:MM:SS"),
    ],
)
def test_pool_start_refuses_a_pool_it_cannot_size(tmp_path, settings, why):
    # The cluster's share has no default; a worker job that ends before the pool is not replaced,
    # so the pool would shrink under a sweep.
    repo = _checkout(tmp_path)
    _pool_settings(repo, {k: v for k, v in _SIZED.items() if k not in settings})
    stub, env = _stubs(tmp_path)
    run = _ctl(repo, env, "start", **settings)
    assert run.returncode == 1 and why in run.stderr
    assert not _calls(stub, "sbatch")


def test_pool_settings_are_read_never_run(tmp_path):
    # slurm/fit_pool.env holds settings: a line that is anything else is refused, not executed.
    repo = _checkout(tmp_path)
    _pool_settings(repo, extra="touch pwned\n")
    stub, env = _stubs(tmp_path)
    run = _ctl(repo, env, "start")
    assert run.returncode == 1 and "is not FITPOOL_NAME=value" in run.stderr
    assert not (repo / "pwned").exists() and not _calls(stub, "sbatch")


def test_pool_start_leaves_a_running_pool_alone(tmp_path):
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-app\n")
    run = _ctl(repo, env, "start")
    assert run.returncode == 0 and "already running as job 100" in run.stdout
    assert not _calls(stub, "sbatch")


def test_pool_start_refuses_an_earlier_pools_workers(tmp_path):
    # Worker jobs a crashed pool left hold the QOS, and no new pool will cancel them.
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="101 fitpool-app-worker\n")
    run = _ctl(repo, env, "start")
    assert run.returncode == 1 and "stop --force" in run.stderr
    assert not _calls(stub, "sbatch")


def test_pool_start_ignores_another_checkouts_pool(tmp_path):
    # Each application runs its own pool from its own pin: one must not find another's.
    repo = _checkout(tmp_path)
    _pool_settings(repo)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-other\n101 fitpool-other-worker\n")
    assert _ctl(repo, env, "start").returncode == 0
    assert _calls(stub, "sbatch")


def test_pool_stop_drains_through_sigterm_and_waits_for_its_workers(tmp_path):
    # Check 11's `stop` with nothing processing: SIGTERM to the pool's batch step only, and the
    # stop returns once the pool has cancelled its worker jobs.
    repo = _checkout(tmp_path)
    stub, env = _stubs(
        tmp_path, jobs="100 fitpool-app\n101 fitpool-app-worker\n102 fitpool-app-worker\n"
    )
    run = _ctl(repo, env, "stop")
    assert run.returncode == 0, run.stderr
    assert "its worker jobs are gone" in run.stdout
    assert _calls(stub, "scancel") == ["scancel --signal=TERM --batch 100"]


def test_pool_stop_reports_worker_jobs_the_pool_left(tmp_path):
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-app\n101 fitpool-app-worker\n")
    (stub / "leaves_workers").touch()
    run = _ctl(repo, env, "stop", FITPOOL_STOP_TIMEOUT="0")
    assert run.returncode == 1 and "worker jobs remain (101)" in run.stderr
    assert _calls(stub, "scancel") == ["scancel --signal=TERM --batch 100"]


def test_pool_stop_hard_cancels_a_pool_that_will_not_end(tmp_path):
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-app\n101 fitpool-app-worker\n")
    (stub / "hangs").touch()
    run = _ctl(repo, env, "stop", FITPOOL_STOP_TIMEOUT="0")
    assert run.returncode == 1 and "hard-cancelled (100 101)" in run.stderr
    assert _calls(stub, "scancel")[-1] == "scancel 100 101"


def test_pool_stop_force_cancels_the_pool_and_its_workers_at_once(tmp_path):
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-app\n101 fitpool-app-worker\n")
    (repo / "slurm" / "fit_pool").mkdir(parents=True)
    (repo / "slurm" / "fit_pool" / "address.txt").write_text("tcp://node007:8786\n")
    run = _ctl(repo, env, "stop", "--force")
    assert run.returncode == 0
    assert _calls(stub, "scancel") == ["scancel 100 101"]
    assert not (repo / "slurm" / "fit_pool" / "address.txt").exists()


def test_pool_status_says_whether_the_address_answers(tmp_path):
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path, jobs="100 fitpool-app\n")
    with _live_pool(repo, {**_POOL_ENV, "NODE": "node007.cluster.example.org"}) as address:
        run = _ctl(repo, env, "status")
    assert f"address:    {address} (reachable)" in run.stdout
    assert f"build root: {repo}/slurm/fit_pool/build" in run.stdout
    assert "FITPOOL_NODELIST=node007 " in run.stdout
    (repo / "slurm" / "fit_pool" / "address.txt").write_text(_dead_address())
    assert "NOT reachable" in _ctl(repo, env, "status").stdout


# --------------------------------------------------------------------------- the pool's job


def _pool_job(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Run fit_pool_server.sh as its job would, with the ctl's defaults, against a stub
    ``petab-mcp-pool`` that records its arguments and environment. The checkout and its dump."""
    repo = _checkout(tmp_path)
    stub, env = _stubs(tmp_path)
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    pool = prefix / "bin" / "petab-mcp-pool"
    pool.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$STUB/pool.argv"\nenv > "$STUB/pool.env"\n'
    )
    pool.chmod(0o755)
    settings = {
        "FITPOOL_PORT": "8786",
        "FITPOOL_MAX_JOBS": "12",
        "FITPOOL_MIN_JOBS": "0",
        "FITPOOL_PARTITION": "compute",
        "FITPOOL_CPUS": "2",
        "FITPOOL_WORKER_JOB": "fitpool-app-worker",
        "FITPOOL_WORKER_CORES": "100",
        "FITPOOL_WORKER_MEM": "200GiB",
        "FITPOOL_WORKER_TIME": "3-00:00:00",
        "FITPOOL_EXCLUDE": _EXCLUDE,
    }
    run = subprocess.run(
        ["bash", str(SCRIPTS / "fit_pool_server.sh")],
        env={**env, **settings, "REPO_ROOT": str(repo), "ENV_PREFIX": str(prefix)},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    dumped = dict(
        line.split("=", 1) for line in (stub / "pool.env").read_text().splitlines() if "=" in line
    )
    return repo, dumped, stub


def test_pool_job_runs_petab_mcp_pool_on_its_published_address(tmp_path):
    repo, _, stub = _pool_job(tmp_path)
    pool_dir = repo / "slurm" / "fit_pool"
    assert (stub / "pool.argv").read_text().split() == [
        "--slurm-max-jobs", "12", "--slurm-min-jobs", "0", "--port", "8786",
        "--address-file", str(pool_dir / "address.txt"),
    ]  # fmt: skip
    assert (pool_dir / "build").is_dir()
    written = S._read_env(pool_dir / "pool.env")
    assert {k: written[k] for k in _POOL_ENV} == _POOL_ENV, "submit's split of a cap reads these"
    assert written["BUILD_ROOT"] == str(pool_dir / "build")


def test_pool_jobs_workers_go_where_the_pool_went(tmp_path):
    # Slurm lets an inherited SBATCH_PARTITION outrank the partition a job script states, so the
    # pool's partition reaches its workers' sbatch as that variable.
    _, dumped, _ = _pool_job(tmp_path)
    assert dumped["SBATCH_PARTITION"] == "compute"
    assert "DASK_JOBQUEUE__SLURM__QUEUE" not in dumped


def test_pool_jobs_workers_are_configured_as_measured(tmp_path):
    # The worker job script dask-jobqueue generates from the job's DASK_* settings: one
    # single-threaded worker a core, death-timeout 600, a walltime past the pool's and no
    # --lifetime, off the excluded nodes, in the node's scratch where the cluster has one; and the
    # scheduler's worker-ttl at 10 minutes (PEtab-MCP#241).
    repo, dumped, _ = _pool_job(tmp_path)
    dask_env = {k: v for k, v in dumped.items() if k.startswith("DASK_")}
    probe = textwrap.dedent(
        """
        import json, dask
        from dask_jobqueue.slurm import SLURMJob
        script = SLURMJob(scheduler="tcp://node007:8786", name="w").job_script()
        print(json.dumps({"script": script,
                          "ttl": dask.config.get("distributed.scheduler.worker-ttl")}))
        """
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("DASK_")}
    run = subprocess.run(
        [sys.executable, "-c", probe],
        env={**clean, **dask_env},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    script = got["script"]
    for line in (
        "#SBATCH -J fitpool-app-worker",
        "#SBATCH --cpus-per-task=100",
        "#SBATCH -t 3-00:00:00",
        f"#SBATCH --exclude={_EXCLUDE}",
        f"#SBATCH -o {repo}/slurm/fit_pool/logs/fitpool-app-worker-%J.out",
    ):
        assert line in script
    assert "--nworkers 100" in script and "--nthreads 1" in script
    assert "--death-timeout 600" in script
    assert "--local-directory ${TMPDIR:-/tmp}" in script
    assert "--lifetime" not in script
    assert got["ttl"] == "10 minutes"


# --------------------------------------------------------------------------- the array task


def _task(
    tmp_path: Path, repo: Path, effort: str | None = "max", **extra: str
) -> subprocess.CompletedProcess:
    """One array task of a 1-combo deepagents sweep of the checkout's toy example."""
    _, env = _stubs(tmp_path)
    combos = repo / "slurm" / "runs" / "combos.tsv"
    combos.parent.mkdir(parents=True, exist_ok=True)
    S.write_combos_file(
        S.build_combos(
            repo / "examples",
            ["toy"],
            ["glm-flash"],
            [1.0],
            effort,
            1,
            "deepagents",
            100,
            500,
            900.0,
        ),
        combos,
    )
    return subprocess.run(
        ["bash", str(SCRIPTS / "array_task.sh")],
        env={
            **env,
            "REPO_ROOT": str(repo),
            "ENV_PREFIX": sys.prefix,
            "COMBOS_FILE": str(combos),
            "RESULTS_DIR": str(repo / "slurm" / "runs" / "results"),
            "FITPOOL_DIR": str(repo / "slurm" / "fit_pool"),
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "77",
            "SLURM_CPUS_PER_TASK": "8",
            "PHOENIX_ENABLED": "0",
            **extra,
        },
        capture_output=True,
        text=True,
    )


def test_task_without_a_pool_stops_before_it_runs(tmp_path):
    repo = _checkout(tmp_path)
    run = _task(tmp_path, repo)
    assert run.returncode == 78 and "fit-pool start" in run.stderr
    assert not (repo / "slurm" / "runs" / "work").exists(), (
        "nothing is staged for a task that cannot fit"
    )


def test_task_with_a_dead_pool_stops_before_it_runs(tmp_path):
    repo = _checkout(tmp_path)
    (repo / "slurm" / "fit_pool").mkdir(parents=True)
    (repo / "slurm" / "fit_pool" / "address.txt").write_text(_dead_address())
    run = _task(tmp_path, repo)
    assert run.returncode == 78 and "not reachable" in run.stderr


def test_task_joins_the_pool_and_builds_in_its_build_root(tmp_path):
    # The task reads the address at its start, exports it and the pool's build root to its run,
    # and pins no fit workers: its starts run on the pool, not on its 8 CPUs.
    repo = _checkout(tmp_path)
    with _live_pool(repo) as address:
        run = _task(tmp_path, repo)
    assert run.returncode == 0, run.stderr
    results = repo / "slurm" / "runs" / "results" / "77_0"
    result = json.loads((results / "result.json").read_text())
    assert result["env"]["PETAB_MCP_FIT_POOL"] == address
    assert result["env"]["PETAB_MCP_BUILD_ROOT"] == str(repo / "slurm" / "fit_pool" / "build")
    assert "PETAB_MCP_FIT_N_WORKERS" not in result["env"]
    assert result["env"]["AMICI_PARALLEL_COMPILE"] == "8"
    assert result["argv"][result["argv"].index("--reasoning-effort") + 1] == "max"
    combo = (results / "combo.env").read_text()
    assert f"PETAB_MCP_FIT_POOL={address}" in combo


# --------------------------------------------------------------------------- python -m gemot.sweep


def test_sweep_refuses_to_run_outside_a_checkout(tmp_path):
    # Every script writes under the checkout's slurm/: run from anywhere else, it would scatter state.
    run = subprocess.run(
        [sys.executable, "-m", "gemot.sweep", "fit-pool", "status"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SRC), "PETAB_MCP_REPO_ROOT": ""},
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "not an application checkout" in run.stderr


def test_sweep_runs_a_service_for_the_checkout_it_is_run_from(tmp_path):
    repo = _checkout(tmp_path)
    _, env = _stubs(tmp_path)
    run = subprocess.run(
        [sys.executable, "-m", "gemot.sweep", "fit-pool", "status"],
        cwd=repo,
        env={**env, "PETAB_MCP_REPO_ROOT": ""},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert f"build root: {repo.resolve()}/slurm/fit_pool/build" in run.stdout


def test_task_leaves_the_effort_to_run_py_unless_the_sweep_set_one(tmp_path):
    # run.py's own default differs by driver (Codex refuses `max`), so a sweep that sets no effort
    # passes none.
    repo = _checkout(tmp_path)
    with _live_pool(repo):
        run = _task(tmp_path, repo, effort=None)
    assert run.returncode == 0, run.stderr
    result = json.loads((repo / "slurm" / "runs" / "results" / "77_0" / "result.json").read_text())
    assert "--reasoning-effort" not in result["argv"]
