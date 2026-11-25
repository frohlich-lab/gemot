#!/usr/bin/env bash
# Long-running Slurm job hosting a persistent, self-hosted Arize Phoenix collector, so many
# sweep array tasks (and many sweeps over time) all export traces to ONE server instead of
# each `run.py` launching its own throwaway in-process app (which only lives as long as that
# one process and isn't reachable from any other node). Submit/stop/inspect it via
# phoenix_ctl.sh (`python -m gemot.sweep phoenix`) -- do not run `phoenix serve` on the login
# node; it holds a compute-node allocation for as long as the service should stay up.
#
# Discovery: once listening, this writes the endpoint to
# $REPO_ROOT/slurm/phoenix_data/endpoint.txt as `http://<node>:<port>` (the node is whatever
# host Slurm scheduled this job on, so it can't be known until the job starts).
# `python -m gemot.sweep submit` reads that file and exports it to every task as
# PHOENIX_COLLECTOR_ENDPOINT -- the env var run.py/tracing.py already honours for "a Phoenix
# collector is already running" (see gemot/tracing.py). Traces persist across restarts in
# $REPO_ROOT/slurm/phoenix_data, so stop/restart (a time-limit hit, or `stop` + `start`) doesn't
# lose history.
set -euo pipefail

: "${REPO_ROOT:?set by phoenix_ctl.sh}"
: "${ENV_PREFIX:?set by phoenix_ctl.sh}"
PORT="${PHOENIX_PORT:-6006}"
# No PHOENIX_WORKING_DIR override, as in phoenix_ctl.sh: this job inherits the .phoenix/ that
# importing gemot set in the shell that submitted it.
DATA_DIR="$REPO_ROOT/slurm/phoenix_data"
mkdir -p "$DATA_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export PHOENIX_WORKING_DIR="$DATA_DIR"

HOST="$(hostname -f 2>/dev/null || hostname)"
ENDPOINT="http://${HOST}:${PORT}"
ENDPOINT_FILE="$DATA_DIR/endpoint.txt"

cleanup() {
    echo "[phoenix_server] shutting down"
    rm -f "$ENDPOINT_FILE"
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup TERM INT EXIT

echo "[phoenix_server] starting on ${HOST}:${PORT}, data dir ${DATA_DIR}"
# PHOENIX_HOST, not `--host`: Phoenix 21 removed the flag (`unrecognized arguments`), and the
# variable works on 20 as well. Binding 0.0.0.0 is what lets array tasks on other nodes reach
# this collector, and from 21 a non-loopback bind turns authentication ON by default: every
# task's span export then gets a 401 (no task holds an API key) while the port still answers,
# so array_task.sh's checks pass and the sweep traces into nothing. PHOENIX_ENABLE_AUTH=false
# keeps the collector as it was on 20, unauthenticated inside the cluster network. (gemot's
# `harness` extra is on Phoenix 21.)
PHOENIX_HOST=0.0.0.0 PHOENIX_ENABLE_AUTH=false phoenix serve --port "$PORT" &
SERVER_PID=$!

# Publish the endpoint only once it's actually accepting connections, so array_task.sh never
# reads a stale/half-started endpoint.
# Readiness wait, in seconds. Generous by default because STARTUP SCALES WITH THE STORE: Phoenix
# runs Alembic migrations and recovers the WAL on open, and at 11.4 GB + a 1.77 GB WAL that took
# longer than the original 120s -- the probe gave up, the endpoint went unpublished and the
# collector exited, taking tracing down for the next sweep. The wait costs nothing when startup
# is fast (it breaks on the first successful connect); override with PHOENIX_READY_TIMEOUT.
READY_TIMEOUT="${PHOENIX_READY_TIMEOUT:-900}"
ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[phoenix_server] server process exited early; see this job's log" >&2
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
    # (array tasks would tunnel to a dead port); fail loudly so the readiness timeout is visible.
    echo "[phoenix_server] server did not accept connections within ${READY_TIMEOUT}s; not publishing $ENDPOINT_FILE" >&2
    exit 1
fi

echo "$ENDPOINT" > "$ENDPOINT_FILE"
echo "[phoenix_server] listening -> $ENDPOINT (published to $ENDPOINT_FILE)"
echo "[phoenix_server] browse from your laptop with:"
echo "    ssh -L ${PORT}:${HOST}:${PORT} <you>@<login-node>"
echo "  then open http://localhost:${PORT}"

# Forward SIGTERM and WAIT: Phoenix inserts spans ASYNCHRONOUSLY behind its HTTP 200s, and on a
# large store the insert queue lags hours behind accepts. A SIGKILL (bare scancel) destroys that
# in-memory backlog — measured 2026-08-16: an entire sweep's lucarelli/bachmann traces were lost
# this way while every POST had returned 200 OK. On SIGTERM Phoenix shuts down its receivers and
# drains what it can; the trap makes the batch script survive long enough for that instead of
# dying with its child. Pair with phoenix_ctl.sh stop, which quiesce-checks BEFORE signalling.
# Unpublish FIRST, then forward. The previous `trap cleanup TERM` removed the endpoint file;
# replacing it with a forward-only handler left a stale endpoint readable for the whole drain
# (up to an hour), so a task starting in that window would export a collector that is shutting
# down. Matters most on the SLURM time-limit path, where nothing else deletes the file.
trap 'echo "[phoenix_server] SIGTERM -> unpublishing endpoint, shutting phoenix down gracefully"; rm -f "$ENDPOINT_FILE"; kill -TERM "$SERVER_PID" 2>/dev/null' TERM
# LOOP the wait. Bash interrupts a blocked `wait` to run the handler and returns 143 WITHOUT
# reaping the child, so a single `wait` falls straight through and the batch step ends while
# phoenix is still draining — SLURM then SIGKILLs it after KillWait (~30s), which is no better
# than the bare scancel this exists to replace. Re-wait until the process is actually gone.
while kill -0 "$SERVER_PID" 2>/dev/null; do
    wait "$SERVER_PID" || true
done
echo "[phoenix_server] phoenix exited"
