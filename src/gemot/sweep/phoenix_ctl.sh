#!/usr/bin/env bash
# Start/stop/inspect the shared Phoenix collector service job (phoenix_server.sh).
#
# Usage, through `python -m gemot.sweep phoenix`, which sets REPO_ROOT and ENV_PREFIX:
#   start    # submit the service job (no-op if already running)
#   status   # show the job + published endpoint
#   stop     # drain and stop it (stop --force: cancel it at once)
set -euo pipefail

: "${REPO_ROOT:?run through python -m gemot.sweep}"
: "${ENV_PREFIX:?run through python -m gemot.sweep}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The one place phoenix_server.sh keeps the store and publishes the endpoint, with no
# PHOENIX_WORKING_DIR override: importing gemot (which runs this script) sets that variable to the
# checkout's .phoenix/, so an override read from it would always send `stop` there.
DATA_DIR="$REPO_ROOT/slurm/phoenix_data"
JOBNAME="petab-phoenix"

cmd="${1:-status}"

case "$cmd" in
    start)
        existing="$(squeue --me --noheader --name="$JOBNAME" --format=%A | head -1)"
        if [[ -n "$existing" ]]; then
            # A job with NO published endpoint is either still starting or DRAINING under a
            # graceful stop (which can take up to an hour). Reporting either as "already
            # running" left sweeps with no endpoint — they then trace to a throwaway in-process
            # Phoenix that dies with the task, which is how a behaviour-measuring sweep ends up
            # with no traces at all. Say which, and fail loudly rather than pretend.
            if [[ -f "$DATA_DIR/endpoint.txt" ]]; then
                echo "[phoenix_ctl] already running as job $existing ($(cat "$DATA_DIR/endpoint.txt"))"
                exit 0
            fi
            # No endpoint yet is NORMAL for a while: readiness scales with the store and the
            # probe now allows 900s, so failing here would abort callers under `set -e` and
            # invite an operator to --force-kill a healthy collector seconds from publishing.
            # A DRAINING job is the case worth calling out, and `stop` leaves a marker for it.
            if [[ -f "$DATA_DIR/.stopping" ]]; then
                echo "[phoenix_ctl] job $existing is DRAINING under a graceful stop — wait for it" \
                     "to exit before starting a new collector, and do not launch a sweep now:" \
                     "it would trace to a throwaway in-process Phoenix." >&2
                exit 1
            fi
            echo "[phoenix_ctl] already running as job $existing; endpoint not published yet" \
                 "(startup scales with the store, up to PHOENIX_READY_TIMEOUT). Poll with" \
                 "'python -m gemot.sweep phoenix status' before launching a sweep."
            exit 0
        fi
        mkdir -p "$REPO_ROOT/slurm/logs"
        # PHOENIX_NODELIST pins the restart to a specific node. Array tasks capture the
        # endpoint URL *once*, at task start (array_task.sh), so if the collector dies
        # mid-sweep -- an OOM kill under many tasks' write load is the usual way -- coming
        # back on a different host strands every in-flight task on a dead URL. Restarting on
        # the same node:port lets their OTel exporters simply reconnect and resume.
        # No partition unless PHOENIX_PARTITION names one: Slurm's own (SBATCH_PARTITION, else the
        # cluster's default) otherwise.
        pin=()
        [[ -n "${PHOENIX_NODELIST:-}" ]] && pin=(--nodelist="$PHOENIX_NODELIST")
        [[ -n "${PHOENIX_PARTITION:-}" ]] && pin+=(--partition="$PHOENIX_PARTITION")
        jobid=$(sbatch --parsable \
            --job-name="$JOBNAME" \
            --cpus-per-task="${PHOENIX_CPUS:-8}" \
            --mem="${PHOENIX_MEM:-64G}" \
            --time="${PHOENIX_TIME:-7-00:00:00}" \
            ${pin[@]+"${pin[@]}"} \
            -o "$REPO_ROOT/slurm/logs/phoenix_server_%j.out" \
            "$SCRIPTS/phoenix_server.sh")
        echo "[phoenix_ctl] submitted job $jobid; poll with: python -m gemot.sweep phoenix status"
        ;;
    stop)
        # GRACEFUL BY DEFAULT. Phoenix acknowledges span POSTs before inserting them (async
        # queue), and inserts into a large store lag far behind accepts — a bare scancel
        # (SIGKILL) destroys the in-memory backlog. Measured 2026-08-16: a sweep's entire
        # lucarelli/bachmann trace set was lost to exactly that, with every POST 200 OK.
        # Sequence here: (1) unpublish the endpoint so nothing new connects; (2) WAIT until the
        # store quiesces (db+wal untouched for QUIESCE_S — the only externally visible signal
        # that the insert queue has drained); (3) SIGTERM only, and wait for the job to end on
        # its own; (4) hard-cancel strictly as a bounded fallback. `stop --force` skips straight
        # to scancel for the cases where losing the backlog is intended (e.g. the store itself
        # is being discarded).
        ids="$(squeue --me --noheader --name="$JOBNAME" --format=%A)"
        if [[ -z "$ids" ]]; then
            echo "[phoenix_ctl] not running"
            rm -f "$DATA_DIR/endpoint.txt"
            exit 0
        fi
        rm -f "$DATA_DIR/endpoint.txt"
        : > "$DATA_DIR/.stopping"          # marker: `start` must not call this job healthy
        trap 'rm -f "$DATA_DIR/.stopping"' EXIT
        if [[ "${2:-}" == "--force" ]]; then
            echo "$ids" | xargs -r -n1 scancel
            echo "[phoenix_ctl] force-stopped ($ids) — any unflushed span backlog is LOST"
            exit 0
        fi
        QUIESCE_S="${PHOENIX_QUIESCE_S:-120}"
        DRAIN_TIMEOUT_S="${PHOENIX_DRAIN_TIMEOUT_S:-3600}"
        # NEVER let this abort the stop: `stat` exits 1 if ANY operand is missing (no db yet, a
        # cleanly-checkpointed store with no -wal, or a PHOENIX_WORKING_DIR pointing elsewhere),
        # and under `set -euo pipefail` that killed the script AFTER endpoint.txt was deleted and
        # BEFORE any signal — orphaning a running collector with no published endpoint.
        stamp() { stat -c '%Y %s' "$DATA_DIR/phoenix.db" "$DATA_DIR/phoenix.db-wal" 2>/dev/null | tr '\n' ' ' || true; }
        echo "[phoenix_ctl] waiting for the store to quiesce (no db/wal change for ${QUIESCE_S}s, timeout ${DRAIN_TIMEOUT_S}s)..."
        waited=0; last="$(stamp)"; still=0
        while (( waited < DRAIN_TIMEOUT_S )); do
            sleep 15; waited=$((waited+15))
            cur="$(stamp)"
            if [[ "$cur" == "$last" ]]; then
                still=$((still+15))
                (( still >= QUIESCE_S )) && break
            else
                still=0; last="$cur"
                echo "[phoenix_ctl]   still draining (${waited}s)..."
            fi
        done
        if (( still < QUIESCE_S )); then
            echo "[phoenix_ctl] WARNING: store never quiesced within ${DRAIN_TIMEOUT_S}s;"                  "proceeding with SIGTERM anyway — some backlog may be lost" >&2
        fi
        echo "$ids" | xargs -r -n1 scancel --signal=TERM --batch
        echo "[phoenix_ctl] SIGTERM sent; waiting up to 300s for a clean exit..."
        for _ in $(seq 1 60); do
            [[ -z "$(squeue --me --noheader --name="$JOBNAME" --format=%A)" ]] && { echo "[phoenix_ctl] stopped cleanly ($ids)"; exit 0; }
            sleep 5
        done
        echo "$ids" | xargs -r -n1 scancel
        echo "[phoenix_ctl] did not exit after SIGTERM; hard-cancelled ($ids)" >&2
        ;;
    status)
        squeue --me --name="$JOBNAME"
        if [[ -f "$DATA_DIR/endpoint.txt" ]]; then
            echo "endpoint: $(cat "$DATA_DIR/endpoint.txt")"
        else
            echo "endpoint: (not published yet -- job may still be starting, or isn't running)"
        fi
        ;;
    *)
        echo "usage: python -m gemot.sweep phoenix {start|status|stop [--force]}" >&2
        exit 1
        ;;
esac
