#!/usr/bin/env bash
# Start, inspect and stop the fit pool's service job (fit_pool_server.sh), as phoenix_ctl.sh does the
# collector's. It only submits, finds and signals the job: petab-mcp-pool publishes the address,
# and drains on SIGTERM.
#
# Usage, through `python -m gemot.sweep fit-pool`, which sets REPO_ROOT and ENV_PREFIX:
#   start          submit the pool's job, unless one is running. FITPOOL_NODELIST=<node> pins it to
#                  a node: the old job's, to restart the pool at the same address (`status` prints
#                  it). Runs already running survive that once PEtab-MCP#216's last item lands.
#   status         the pool's job and its worker jobs, its address and its build root
#   stop           SIGTERM: the pool unpublishes its address, waits (at most 600 s) until no batch is
#                  processing, and cancels its worker jobs. This waits for both.
#   stop --force   cancel the pool's job and its worker jobs at once, failing every fit in flight
#
# The pool's settings are FITPOOL_* variables, read from the environment and then from the
# checkout's slurm/fit_pool.env (one NAME=value a line, `#` comments), so a variable given on the
# command line wins over the application's file. What depends on the cluster has no default:
#   FITPOOL_MAX_JOBS                               worker jobs while batches wait for a worker
#   FITPOOL_WORKER_CORES                           workers per job, one single-threaded worker a core
#   FITPOOL_TIME                                   the pool's own walltime, past every sweep it serves
#                                                  and within the partition's limit
#   FITPOOL_PARTITION, FITPOOL_EXCLUDE, FITPOOL_NODELIST   none, so Slurm's own (SBATCH_PARTITION...)
# What the pool needs anywhere has one (PEtab-MCP#241, measured at 1200 workers and 100 servers):
#   FITPOOL_MIN_JOBS (0)                           worker jobs kept while idle
#   FITPOOL_WORKER_MEM (2 GiB a worker)            per worker job
#   FITPOOL_WORKER_TIME (FITPOOL_TIME)             no shorter than the pool's: see `start`
#   FITPOOL_CPUS (2), FITPOOL_MEM (8G)             the pool's own job: its scheduler used about one
#                                                  core and 1.4 GB
#   FITPOOL_PORT (8786), FITPOOL_STOP_TIMEOUT (900 s that `stop` waits for the job to end)
# `submit` reads them back from the pool (slurm/fit_pool/pool.env) to split a per-user CPU cap with
# the tasks.
set -euo pipefail

: "${REPO_ROOT:?run through python -m gemot.sweep fit-pool}"
: "${ENV_PREFIX:?run through python -m gemot.sweep fit-pool}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_DIR="$REPO_ROOT/slurm/fit_pool"
ADDRESS_FILE="$POOL_DIR/address.txt"
# Per checkout: each application runs a pool of its own, from its own pin (a server refuses a pool
# of another environment), so one must not find, or stop, another's.
JOBNAME="fitpool-${REPO_ROOT##*/}"
WORKER_JOB="$JOBNAME-worker"
SETTINGS="$REPO_ROOT/slurm/fit_pool.env"

refuse() { echo "[fit_pool_ctl] $*" >&2; exit 1; }

# The application's settings, under any given in the environment. Read, not sourced: a line is a
# setting, never a command.
if [[ -f "$SETTINGS" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        key="${line%%=*}"
        [[ "$line" == *=* && "$key" =~ ^FITPOOL_[A-Z_]+$ ]] \
            || refuse "$SETTINGS: '$line' is not FITPOOL_NAME=value."
        [[ -n "${!key+set}" ]] || export "$key=${line#*=}"
    done < "$SETTINGS"
fi

# The ids of my jobs of one name, on one line.
jobs_named() { squeue --me --noheader --name="$1" --format=%A | tr '\n' ' ' | sed 's/ *$//'; }

# A Slurm walltime (D-HH:MM:SS or HH:MM:SS) in seconds; nothing when it is neither.
seconds() {
    local spec="$1" days=0 h m s
    if [[ "$spec" == *-* ]]; then days="${spec%%-*}"; spec="${spec#*-}"; fi
    IFS=: read -r h m s <<< "$spec"
    [[ "$days" =~ ^[0-9]+$ && "$h" =~ ^[0-9]+$ && "$m" =~ ^[0-9]+$ && "$s" =~ ^[0-9]+$ ]] || return 0
    echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

cmd="${1:-status}"

case "$cmd" in
    start)
        existing="$(jobs_named "$JOBNAME")"
        if [[ -n "$existing" ]]; then
            if [[ -f "$ADDRESS_FILE" ]]; then
                echo "[fit_pool_ctl] already running as job $existing ($(cat "$ADDRESS_FILE"))"
            else
                echo "[fit_pool_ctl] already running as job $existing; its address is not published" \
                     "yet (starting, or draining under a stop). Poll with 'python -m gemot.sweep" \
                     "fit-pool status' before submitting a sweep."
            fi
            exit 0
        fi
        orphans="$(jobs_named "$WORKER_JOB")"
        [[ -z "$orphans" ]] || refuse "worker jobs of an earlier pool are still queued or running" \
            "($orphans), holding their CPUs, and no pool will cancel them. Cancel them with" \
            "'python -m gemot.sweep fit-pool stop --force', then start."

        missing=""
        for knob in FITPOOL_MAX_JOBS FITPOOL_WORKER_CORES FITPOOL_TIME; do
            [[ -n "${!knob:-}" ]] || missing="$missing $knob"
        done
        [[ -z "$missing" ]] || refuse "set$missing for this cluster, in the environment or in" \
            "$SETTINGS: how many worker jobs of how many cores, and how long the pool runs."
        export FITPOOL_CPUS="${FITPOOL_CPUS:-2}"
        export FITPOOL_MEM="${FITPOOL_MEM:-8G}"
        export FITPOOL_PORT="${FITPOOL_PORT:-8786}"
        export FITPOOL_MIN_JOBS="${FITPOOL_MIN_JOBS:-0}"
        export FITPOOL_WORKER_TIME="${FITPOOL_WORKER_TIME:-$FITPOOL_TIME}"
        export FITPOOL_PARTITION="${FITPOOL_PARTITION:-}"
        export FITPOOL_EXCLUDE="${FITPOOL_EXCLUDE:-}"
        export FITPOOL_WORKER_JOB="$WORKER_JOB"

        for knob in FITPOOL_CPUS FITPOOL_PORT FITPOOL_MAX_JOBS FITPOOL_MIN_JOBS FITPOOL_WORKER_CORES; do
            [[ "${!knob}" =~ ^[0-9]+$ ]] || refuse "$knob=${!knob} is not a whole number."
        done
        (( FITPOOL_MAX_JOBS >= 1 && FITPOOL_MIN_JOBS <= FITPOOL_MAX_JOBS )) \
            || refuse "the pool needs 0 <= FITPOOL_MIN_JOBS ($FITPOOL_MIN_JOBS) <= FITPOOL_MAX_JOBS" \
                      "($FITPOOL_MAX_JOBS), and at least one worker job."
        (( FITPOOL_WORKER_CORES >= 1 )) || refuse "FITPOOL_WORKER_CORES must be at least 1."
        export FITPOOL_WORKER_MEM="${FITPOOL_WORKER_MEM:-$(( 2 * FITPOOL_WORKER_CORES ))GiB}"
        for knob in FITPOOL_TIME FITPOOL_WORKER_TIME; do
            [[ -n "$(seconds "${!knob}")" ]] || refuse "$knob=${!knob} is not D-HH:MM:SS or HH:MM:SS."
        done
        (( $(seconds "$FITPOOL_WORKER_TIME") >= $(seconds "$FITPOOL_TIME") )) \
            || refuse "FITPOOL_WORKER_TIME=$FITPOOL_WORKER_TIME ends before the pool's" \
                      "FITPOOL_TIME=$FITPOOL_TIME. A worker job that ends is never replaced (jobqueue's" \
                      "adapt does not resubmit a job of several workers), so the pool would shrink."

        mkdir -p "$REPO_ROOT/slurm/logs" "$POOL_DIR"
        rm -f "$ADDRESS_FILE"   # a crashed pool's; the new one writes its own once it listens
        pin=()
        [[ -n "$FITPOOL_PARTITION" ]] && pin+=(--partition="$FITPOOL_PARTITION")
        [[ -n "${FITPOOL_NODELIST:-}" ]] && pin+=(--nodelist="$FITPOOL_NODELIST")
        [[ -n "$FITPOOL_EXCLUDE" ]] && pin+=(--exclude="$FITPOOL_EXCLUDE")
        # --export=ALL carries the settings above to the job: they hold commas (FITPOOL_EXCLUDE),
        # which sbatch's --export=VAR=value list cannot.
        jobid=$(sbatch --parsable \
            --job-name="$JOBNAME" \
            --cpus-per-task="$FITPOOL_CPUS" \
            --mem="$FITPOOL_MEM" \
            --time="$FITPOOL_TIME" \
            ${pin[@]+"${pin[@]}"} \
            --export=ALL \
            -o "$REPO_ROOT/slurm/logs/fit_pool_%j.out" \
            "$SCRIPTS/fit_pool_server.sh")
        echo "[fit_pool_ctl] submitted job $jobid: up to $FITPOOL_MAX_JOBS worker jobs of" \
             "$FITPOOL_WORKER_CORES workers. Poll with: python -m gemot.sweep fit-pool status"
        ;;
    stop)
        ids="$(jobs_named "$JOBNAME")"
        if [[ "${2:-}" == "--force" ]]; then
            all="$(echo "$ids $(jobs_named "$WORKER_JOB")" | xargs)"
            rm -f "$ADDRESS_FILE"
            if [[ -z "$all" ]]; then
                echo "[fit_pool_ctl] not running"
                exit 0
            fi
            # shellcheck disable=SC2086  # one id per word
            scancel $all
            echo "[fit_pool_ctl] force-stopped ($all): every fit in flight on the pool failed"
            exit 0
        fi
        if [[ -z "$ids" ]]; then
            rm -f "$ADDRESS_FILE"   # a crashed pool's, if any
            left="$(jobs_named "$WORKER_JOB")"
            [[ -z "$left" ]] || refuse "not running, but worker jobs of an earlier pool remain" \
                "($left). Cancel them with 'python -m gemot.sweep fit-pool stop --force'."
            echo "[fit_pool_ctl] not running"
            exit 0
        fi
        for id in $ids; do scancel --signal=TERM --batch "$id"; done
        timeout="${FITPOOL_STOP_TIMEOUT:-900}"
        echo "[fit_pool_ctl] SIGTERM sent to $ids: the pool drains (at most 600 s) and cancels its" \
             "worker jobs. Waiting up to ${timeout}s..."
        # Until the pool's job AND its worker jobs are gone: the pool cancels them as it closes, and
        # Slurm lists a cancelled job for a while after.
        waited=0
        while true; do
            pool="$(jobs_named "$JOBNAME")"
            left="$(jobs_named "$WORKER_JOB")"
            [[ -z "$pool$left" ]] && break
            if (( waited >= timeout )); then
                if [[ -n "$pool" ]]; then
                    all="$(echo "$pool $left" | xargs)"
                    # shellcheck disable=SC2086
                    scancel $all
                    rm -f "$ADDRESS_FILE"
                    refuse "the pool's job did not end ${timeout}s after SIGTERM; hard-cancelled" \
                        "($all), failing any fit still on it."
                fi
                refuse "the pool's job ended, but its worker jobs remain ($left). Cancel them" \
                    "with 'python -m gemot.sweep fit-pool stop --force'."
            fi
            sleep 5
            waited=$((waited + 5))
        done
        echo "[fit_pool_ctl] stopped cleanly ($ids); its worker jobs are gone"
        ;;
    status)
        squeue --me --name="$JOBNAME,$WORKER_JOB"
        if [[ -f "$ADDRESS_FILE" ]]; then
            address="$(tr -d '[:space:]' < "$ADDRESS_FILE")"
            if "$ENV_PREFIX/bin/python" -c \
                    'import sys; from gemot.sweep.submit import probe; probe(sys.argv[1])' \
                    "$address" 2>/dev/null; then
                echo "address:    $address (reachable)"
            else
                echo "address:    $address (NOT reachable: the pool died without removing it)"
            fi
        else
            echo "address:    (not published: the job is starting or draining, or isn't running)"
        fi
        echo "build root: $POOL_DIR/build"
        if [[ -f "$POOL_DIR/pool.env" ]]; then
            echo "settings:   $(tr '\n' ' ' < "$POOL_DIR/pool.env")"
            node="$(sed -n 's/^NODE=//p' "$POOL_DIR/pool.env")"
            echo "restart it at the same address: python -m gemot.sweep fit-pool stop, then"
            echo "  FITPOOL_NODELIST=${node%%.*} python -m gemot.sweep fit-pool start"
        fi
        ;;
    *)
        echo "usage: python -m gemot.sweep fit-pool {start|status|stop [--force]}" >&2
        exit 1
        ;;
esac
