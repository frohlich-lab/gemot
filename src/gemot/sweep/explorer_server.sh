#!/usr/bin/env bash
# Long-running Slurm job hosting the run-explorer web UI (`python -m gemot.explorer`), so archived
# runs *and* in-flight sweep tasks can be browsed from one place. Submit/stop/inspect it via
# explorer_ctl.sh (`python -m gemot.sweep explorer`) -- do not run it on the login node; it holds a
# compute-node allocation for as long as the UI should stay up.
#
# Discovery: once listening, this writes the endpoint to
# $REPO_ROOT/slurm/explorer_data/endpoint.txt as `http://<node>:<port>` (the node is whatever
# host Slurm scheduled this job on, so it can't be known until the job starts). Tunnel to that
# host to browse it -- see the message this prints on startup.
#
# Live runs: the explorer discovers in-flight runs by globbing `<examples-root>/*/work/
# run_live.json` -- one level deep. On the cluster each array task runs from its OWN scratch
# copy (`slurm/runs/work/<job>_<task>/examples/<example>/`, see array_task.sh), so no single
# directory holds them all. This script therefore maintains a "live root" of symlinks --
# one per scratch example dir plus the real `examples/<example>` dirs (for local runs) --
# and re-syncs it every $EXPLORER_LIVE_REFRESH seconds so tasks that start later (throttled
# array slots) show up without a restart.
#
# Phoenix: the explorer fetches spans server-side (explorer/bundle.py), so only this port
# needs tunnelling -- not the collector's 6006 as well.
set -euo pipefail

: "${REPO_ROOT:?set by explorer_ctl.sh}"
: "${ENV_PREFIX:?set by explorer_ctl.sh}"
PORT="${EXPLORER_PORT:-8000}"
DATA_DIR="${EXPLORER_DATA_DIR:-$REPO_ROOT/slurm/explorer_data}"
SCRATCH_ROOT="${PETAB_MCP_SCRATCH:-$REPO_ROOT/slurm/runs/work}"
REFRESH="${EXPLORER_LIVE_REFRESH:-30}"
LIVE_ROOT="$DATA_DIR/live_root"
ENDPOINT_FILE="$DATA_DIR/endpoint.txt"
mkdir -p "$LIVE_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Rebuild the live root: one symlink per example dir that could hold an in-flight run.
# The root is rebuilt from scratch each pass rather than appended to: a task that finished (or
# died) since the last pass must lose its link, or a stale marker keeps it pinned in the live
# list forever. Finished runs are served from the archive instead, so dropping the link is the
# hand-off, not a loss.
sync_live_root() {
    local d name task running
    find "$LIVE_ROOT" -maxdepth 1 -type l -delete 2>/dev/null || true
    # A scratch dir is only *live* if its array task is still queued/running. Tasks that were
    # cancelled or hit the time limit die without writing result.json, so their run_live.json
    # marker survives forever -- linking those in would show months-old corpses as "in flight".
    running="$(squeue --me --noheader --format=%i 2>/dev/null || true)"
    for d in "$SCRATCH_ROOT"/*/examples/*/; do
        [[ -d "$d" ]] || continue                      # no matches -> unexpanded glob
        d="${d%/}"
        [[ "$(basename "$d")" == _* ]] && continue     # skip any _-prefixed staging dir
        task="$(basename "$(dirname "$(dirname "$d")")")"   # <job>_<task>
        grep -qx -- "$task" <<<"$running" || continue
        # <work-root>/<job>_<task>/examples/<example> -> <example>_<job>_<task>
        name="$(basename "$d")_$task"
        ln -sfn "$d" "$LIVE_ROOT/$name"
    done
    for d in "$REPO_ROOT"/examples/*/; do              # local (non-Slurm) runs
        d="${d%/}"
        name="$(basename "$d")"
        [[ "$name" == _* ]] && continue                # _runs_archive
        [[ -d "$d/work" ]] || continue
        ln -sfn "$d" "$LIVE_ROOT/$name"
    done
    return 0
}

HOST="$(hostname -f 2>/dev/null || hostname)"
ENDPOINT="http://${HOST}:${PORT}"
PHOENIX_URL="$(cat "$REPO_ROOT/slurm/phoenix_data/endpoint.txt" 2>/dev/null || true)"

cleanup() {
    echo "[explorer_server] shutting down"
    rm -f "$ENDPOINT_FILE"
    [[ -n "${REFRESH_PID:-}" ]] && kill "$REFRESH_PID" 2>/dev/null || true
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup TERM INT EXIT

sync_live_root
if [[ -z "$PHOENIX_URL" ]]; then
    echo "[explorer_server] warning: no Phoenix endpoint published " \
         "($REPO_ROOT/slurm/phoenix_data/endpoint.txt missing) -- the model-variant and " \
         "nllh/BIC panels work from the archive alone, but the conversation panels will be " \
         "empty. Start it with 'python -m gemot.sweep phoenix start', then restart this job." >&2
fi

echo "[explorer_server] starting on ${HOST}:${PORT} | phoenix='${PHOENIX_URL:-<none>}'"
cd "$REPO_ROOT"
PX_ARGS=()
[[ -n "$PHOENIX_URL" ]] && PX_ARGS=(--phoenix "$PHOENIX_URL")
"$ENV_PREFIX/bin/python" -m gemot.explorer \
    --host 0.0.0.0 --port "$PORT" \
    --archive "$REPO_ROOT/examples/_runs_archive" \
    --examples-root "$LIVE_ROOT" \
    "${PX_ARGS[@]}" &
SERVER_PID=$!

# Keep the live root current while the UI is up (new array tasks, finished ones cleaned away).
while true; do sleep "$REFRESH"; sync_live_root || true; done &
REFRESH_PID=$!

# Publish the endpoint only once it's actually accepting connections, so nobody tunnels to a
# half-started server.
ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[explorer_server] server process exited early; see this job's log" >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    # The wait expired without the port ever accepting a connection. Do NOT publish the endpoint
    # (a tunnel would hit a dead port); fail loudly so the readiness timeout is visible.
    echo "[explorer_server] server did not accept connections within 120s; not publishing $ENDPOINT_FILE" >&2
    exit 1
fi

echo "$ENDPOINT" > "$ENDPOINT_FILE"
echo "[explorer_server] listening -> $ENDPOINT (published to $ENDPOINT_FILE)"
echo "[explorer_server] browse from your laptop with:"
echo "    ssh -L ${PORT}:${HOST}:${PORT} <you>@<login-node>"
echo "    # then open http://localhost:${PORT}"
wait "$SERVER_PID"
