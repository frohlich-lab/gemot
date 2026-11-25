#!/usr/bin/env bash
# Start/stop/inspect the run-explorer UI service job (explorer_server.sh).
#
# Usage, through `python -m gemot.sweep explorer`, which sets REPO_ROOT and ENV_PREFIX:
#   start    # submit the service job (no-op if already running)
#   status   # show the job, published endpoint + the ssh tunnel to use
#   stop     # cancel it
set -euo pipefail

: "${REPO_ROOT:?run through python -m gemot.sweep}"
: "${ENV_PREFIX:?run through python -m gemot.sweep}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${EXPLORER_DATA_DIR:-$REPO_ROOT/slurm/explorer_data}"
JOBNAME="petab-explorer"

cmd="${1:-status}"

case "$cmd" in
    start)
        existing="$(squeue --me --noheader --name="$JOBNAME" --format=%A | head -1)"
        if [[ -n "$existing" ]]; then
            echo "[explorer_ctl] already running as job $existing"
            exit 0
        fi
        mkdir -p "$REPO_ROOT/slurm/logs"
        # No partition unless EXPLORER_PARTITION names one: Slurm's own otherwise.
        part=()
        [[ -n "${EXPLORER_PARTITION:-}" ]] && part=(--partition="$EXPLORER_PARTITION")
        jobid=$(sbatch --parsable \
            --job-name="$JOBNAME" \
            ${part[@]+"${part[@]}"} \
            --cpus-per-task="${EXPLORER_CPUS:-2}" \
            --mem="${EXPLORER_MEM:-8G}" \
            --time="${EXPLORER_TIME:-1-00:00:00}" \
            -o "$REPO_ROOT/slurm/logs/explorer_server_%j.out" \
            "$SCRIPTS/explorer_server.sh")
        echo "[explorer_ctl] submitted job $jobid; poll with: python -m gemot.sweep explorer status"
        ;;
    stop)
        ids="$(squeue --me --noheader --name="$JOBNAME" --format=%A)"
        if [[ -z "$ids" ]]; then
            echo "[explorer_ctl] not running"
        else
            echo "$ids" | xargs -r -n1 scancel
            echo "[explorer_ctl] stopped ($ids)"
        fi
        rm -f "$DATA_DIR/endpoint.txt"
        ;;
    status)
        squeue --me --name="$JOBNAME"
        if [[ -f "$DATA_DIR/endpoint.txt" ]]; then
            endpoint="$(cat "$DATA_DIR/endpoint.txt")"
            hostport="${endpoint#http://}"
            echo "endpoint: $endpoint"
            echo "tunnel:   ssh -L ${hostport##*:}:${hostport} <you>@<login-node>"
            echo "          then open http://localhost:${hostport##*:}"
        else
            echo "endpoint: (not published yet -- job may still be starting, or isn't running)"
        fi
        ;;
    *)
        echo "usage: python -m gemot.sweep explorer {start|status|stop}" >&2
        exit 1
        ;;
esac
