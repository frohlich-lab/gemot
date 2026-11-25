#!/usr/bin/env bash
# The fit pool's service job: one long-lived SLURM job running petab-mcp-pool, whose dask scheduler
# every server of a sweep joins (PETAB_MCP_FIT_POOL), and which submits the SLURM worker jobs their
# fits' starts run on. Submitted, found and signalled by fit_pool_ctl.sh (`python -m gemot.sweep
# fit-pool start`) -- do not run it on the login node. petab-mcp-pool does the rest: it publishes
# the address once the scheduler listens, and on SIGTERM unpublishes it, waits (at most 10 min) for
# the batches running, and closes the cluster, cancelling its worker jobs.
#
# EXEC, so the pool IS this job's batch step: `scancel --signal=TERM --batch` (the ctl's `stop`)
# reaches petab-mcp-pool itself. Behind a shell, bash would hold the signal until its child exited.
#
# The worker jobs are set in dask's own config, as DASK_* variables (PEtab-MCP's README, "Fitting on
# SLURM"), from the FITPOOL_* settings the ctl resolved and checked. The cluster's share (how many
# jobs of how many cores, for how long, where) is the operator's; the rest is as PEtab-MCP#241
# measured at 1200 workers and 100 servers:
#   * processes = cores: one single-threaded worker per core, so each start has a core to itself;
#   * a walltime longer than the pool runs, and no --lifetime. A worker job is never replaced when it
#     ends (jobqueue's adapt does not resubmit a job of several workers), and a worker reaching its
#     lifetime drops its running batch as a killed one does. So restart the pool between sweeps;
#   * death-timeout 600: the default 60 s was too short for a full job's workers to import from
#     shared storage;
#   * worker-ttl 10 minutes: at 1200 workers and 100 servers the warm heartbeat gaps stayed under
#     10 s, so this only removes a worker that is really gone.
set -euo pipefail

: "${REPO_ROOT:?set by fit_pool_ctl.sh}"
: "${ENV_PREFIX:?set by fit_pool_ctl.sh}"
for knob in FITPOOL_PORT FITPOOL_MAX_JOBS FITPOOL_MIN_JOBS FITPOOL_CPUS \
            FITPOOL_WORKER_JOB FITPOOL_WORKER_CORES FITPOOL_WORKER_MEM FITPOOL_WORKER_TIME; do
    : "${!knob:?set by fit_pool_ctl.sh}"
done
POOL_DIR="$REPO_ROOT/slurm/fit_pool"
mkdir -p "$POOL_DIR/build" "$POOL_DIR/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# The worker jobs go where this job went. The pool's sbatch reads SBATCH_PARTITION (and the account
# and QOS) from this job's environment, which it inherited from the shell that started it, and an
# input variable outranks a partition the job script states. So a FITPOOL_PARTITION that sent this
# job elsewhere has to be that variable too, or the workers would land in the shell's.
[[ -n "${FITPOOL_PARTITION:-}" ]] && export SBATCH_PARTITION="$FITPOOL_PARTITION"
export DASK_JOBQUEUE__SLURM__NAME="$FITPOOL_WORKER_JOB"
export DASK_JOBQUEUE__SLURM__CORES="$FITPOOL_WORKER_CORES"
export DASK_JOBQUEUE__SLURM__PROCESSES="$FITPOOL_WORKER_CORES"
export DASK_JOBQUEUE__SLURM__MEMORY="$FITPOOL_WORKER_MEM"
export DASK_JOBQUEUE__SLURM__WALLTIME="$FITPOOL_WORKER_TIME"
export DASK_JOBQUEUE__SLURM__DEATH_TIMEOUT=600
# Single-quoted: expanded by each worker job's own shell, to its node's scratch where the cluster
# sets TMPDIR.
export DASK_JOBQUEUE__SLURM__LOCAL_DIRECTORY='${TMPDIR:-/tmp}'
export DASK_JOBQUEUE__SLURM__LOG_DIRECTORY="$POOL_DIR/logs"
if [[ -n "${FITPOOL_EXCLUDE:-}" ]]; then
    export DASK_JOBQUEUE__SLURM__JOB_EXTRA_DIRECTIVES="['--exclude=$FITPOOL_EXCLUDE']"
fi
export DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="10 minutes"

# What this pool is, for `submit` (splitting a per-user CPU cap) and the ctl's `status`. Written before the address,
# so a published address always has it.
{
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "NODE=$(hostname -f 2>/dev/null || hostname)"
    echo "ENV_PREFIX=$ENV_PREFIX"
    echo "BUILD_ROOT=$POOL_DIR/build"
    for knob in FITPOOL_PORT FITPOOL_MAX_JOBS FITPOOL_MIN_JOBS FITPOOL_PARTITION FITPOOL_CPUS \
                FITPOOL_WORKER_JOB FITPOOL_WORKER_CORES FITPOOL_WORKER_MEM FITPOOL_WORKER_TIME \
                FITPOOL_EXCLUDE; do
        echo "$knob=${!knob:-}"
    done
} > "$POOL_DIR/pool.env"
echo "[fit_pool_server] $(tr '\n' ' ' < "$POOL_DIR/pool.env")"

exec "$ENV_PREFIX/bin/petab-mcp-pool" \
    --slurm-max-jobs "$FITPOOL_MAX_JOBS" \
    --slurm-min-jobs "$FITPOOL_MIN_JOBS" \
    --port "$FITPOOL_PORT" \
    --address-file "$POOL_DIR/address.txt"
