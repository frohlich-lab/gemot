#!/usr/bin/env bash
# One Slurm array task: run a single (example, model, temperature, reasoning effort, repeat) combo
# from the sweep's combos file, its fits on the sweep's fit pool. Submitted via sbatch --array=... by
# `python -m gemot.sweep submit` (gemot/sweep/submit.py) -- do not invoke directly outside Slurm (it
# relies on SLURM_ARRAY_TASK_ID / SLURM_ARRAY_JOB_ID and the env vars submit exports with --export).
#
# WHY THE STAGING BELOW: gemot's runner always writes to a FIXED
# ``<example_dir>/work`` and ``<example_dir>/.amici_build`` (see its cleanup() and
# build_root), both derived from ``Path(__file__).resolve().parent`` in the example's run.py.
# Two array tasks running the SAME example concurrently would `rmtree` and recompile on top of
# each other. So each task gets a private copy of its example directory under a per-task
# scratch dir and runs run.py from there.
#
# The files are COPIED, not symlinked: ``Path(__file__).resolve()`` FOLLOWS symlinks, so a
# symlinked run.py would resolve back to the real checkout and put work/ + .amici_build/ there
# again -- defeating the isolation. A real copy makes __file__ resolve inside the scratch dir,
# so work/ and .amici_build/ are created fresh and privately per task. The example dirs are
# tiny (a few TSV/XML/py files), so copying is cheap. The harness itself is NOT staged at all:
# it is an installed package (gemot) resolved from the conda env, not from the scratch
# tree. PETAB_MCP_REPO_ROOT below pins its paths back to the real checkout, so every task's
# result archives to one central examples/_runs_archive/. .git is not staged either: the scratch
# dir lives inside the repo, so the runner's `git rev-parse` (run from the scratch example dir)
# walks up and finds the real repo's .git anyway.
#
# The AMICI build root is NOT the staged .amici_build, though: it is the pool's (see the pool gate
# below), one folder on shared storage for every task of every sweep on that pool. The workers import
# each model from the path the server built it at, so it cannot be private, and runs of one model
# then compile it once. Two servers building one structure wait for each other on the build lock
# there (PEtab-MCP), which is what sharing it needed.
set -euo pipefail

: "${REPO_ROOT:?set by gemot.sweep submit}"
: "${ENV_PREFIX:?set by gemot.sweep submit}"
: "${COMBOS_FILE:?set by gemot.sweep submit}"
: "${RESULTS_DIR:?set by gemot.sweep submit}"
: "${FITPOOL_DIR:?set by gemot.sweep submit}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# combos.tsv: header + one row per array index (0-based -> line TASK_ID+2).
ROW="$(sed -n "$((TASK_ID + 2))p" "$COMBOS_FILE")"
if [[ -z "$ROW" ]]; then
    echo "[array_task] no row for task $TASK_ID in $COMBOS_FILE" >&2
    exit 1
fi
IFS=$'\t' read -r IDX EXAMPLE DRIVER MODEL TEMPERATURE REASONING_EFFORT REPEAT N_STARTS \
    RECURSION_LIMIT MAX_WALL_SECONDS PER_START_TIMEOUT <<< "$ROW"
# The row's own idx must equal this task's index. submit writes a UNIQUE combos file
# per submission, so this should be unreachable -- but it is the check that turns the dangerous
# version of the old bug into a loud one. A second sweep that overwrote a fixed combos.tsv with
# AS MANY rows would have had every task silently run a DIFFERENT combo, and nothing downstream
# (result.json included) would have contradicted it.
if [[ "$IDX" != "$TASK_ID" ]]; then
    echo "[array_task] combos row mismatch: task $TASK_ID read idx=$IDX from $COMBOS_FILE — the "\
         "file changed under this array. Refusing to run a combo that is not this task's." >&2
    exit 1
fi
echo "[array_task] task=$TASK_ID idx=$IDX example=$EXAMPLE driver=$DRIVER model=$MODEL " \
     "temperature=$TEMPERATURE reasoning_effort=$REASONING_EFFORT repeat=$REPEAT n_starts=$N_STARTS " \
     "max_wall_seconds=$MAX_WALL_SECONDS per_start_timeout=${PER_START_TIMEOUT:-default}"

# The Fides per-start cap, PER EXAMPLE: the AGENT_PER_START_TIMEOUT its run.py declares, if any
# (submit reads it). Empty/None leaves petab_mcp's own default (180 s) in place. Each of the fit's
# batches carries it to the pool, so runs with different caps share one pool.
if [[ -n "${PER_START_TIMEOUT:-}" && "$PER_START_TIMEOUT" != "None" ]]; then
    export PETAB_MCP_FIT_PER_START_TIMEOUT="$PER_START_TIMEOUT"
    echo "[array_task] per-start Fides cap: PETAB_MCP_FIT_PER_START_TIMEOUT=$PER_START_TIMEOUT"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
# The env's own compilers (environment.yml's gcc/gxx), for this machine's architecture.
export CC="$ENV_PREFIX/bin/$(uname -m)-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/$(uname -m)-conda-linux-gnu-c++"

# No PETAB_MCP_FIT_N_WORKERS here, deliberately: the fits' starts run on the pool's workers, not on
# this task's CPUs, which is why it no longer holds 100 of them. Pinning it to the allocation was what
# sized a local multistart to the task.

# Compile the generated C++ in parallel. AMICI defaults CMAKE_BUILD_PARALLEL_LEVEL to "1" unless
# AMICI_PARALLEL_COMPILE is set (exporters/sundials/templates/setup.template.py), so every model
# built serially however many cores the task held -- measured at `-j 1` on a 100-core allocation,
# where the stub alone took ~13 minutes. It is not a one-off cost either: every structural change
# is a whole-model re-author, so a run pays a fresh compile per version.
#
# At most 8. Compilation is memory-bound per process and --mem is shared by the whole task, so
# `-j 100` against a 64G allocation risked the OOM killer taking the run instead of saving time,
# when a task held 100 CPUs for its fits. It is also why a task now asks for 8: the server compiles
# every new model version here, on the task's own node (never on a pool worker), and with fewer CPUs
# every compile would be slower than it was. The cap means a small allocation is never
# oversubscribed.
#
# Only the C++ build is worth parallelising. Symbolic codegen -- the step BEFORE it -- has its own
# knob, AMICI_IMPORT_NPROCS, and it is deliberately left at its default of 1: measured on bachmann
# (the largest model here, 89 parameters), codegen is 1.5s serial but 32.6s at 4 procs and 43.3s at
# 8, because AMICI's pool uses `spawn` and each worker re-imports sympy/amici. Identical output,
# 22-29x slower. Do not set it.
_amici_jobs="${SLURM_CPUS_PER_TASK:-1}"
[[ "$_amici_jobs" -gt 8 ]] && _amici_jobs=8
export AMICI_PARALLEL_COMPILE="${AMICI_PARALLEL_COMPILE:-$_amici_jobs}"
echo "[array_task] compile parallelism: AMICI_PARALLEL_COMPILE=$AMICI_PARALLEL_COMPILE"

# STARTUP JITTER. Slurm has no minimum spacing between array-task starts (%N throttles
# concurrency, not timing), so a low-CPU sweep whose whole array fits the queue launches every
# task in the same instant, each cold-importing the AMICI/pyPESTO stack from ONE shared conda env
# on network storage.
#
# CORRECTION 2026-08-15 — the "import storm" this was built for is NOT what kills those tasks, and
# the original rationale here (40 simultaneous starts starved 8 of them) was wrong. Re-checked
# against the week: the startup deaths are ANTI-correlated with concurrency (all 8 tasks of one
# simultaneous wave survived; the ones that died had started alone, minutes apart) and they
# CONCENTRATE ON SPECIFIC NODES across different jobs on different days (13 deaths on four nodes).
# A healthy node imports the stack in ~35s; the failures exceed
# 120s having written zero bytes. That is a stalling mount, not contention. The real mitigations
# are excluding those nodes (submit --exclude) and a less brutal server-start timeout.
# The jitter is KEPT because staggering cold starts off one shared env is harmless and mildly
# considerate to the filesystem — but it is not the fix, and nothing here should be read as
# evidence that it ever was. Deterministic spread, not random: task_id*7s
# wrapped at PETAB_SWEEP_JITTER_WINDOW_S covers the window uniformly, is reproducible per task,
# and re-runs of a single task (array=7) sleep at most once. Skipped for small arrays — under
# 16 concurrent starts the storm never materialised across ~30 prior sweeps.
_JITTER_WINDOW="${PETAB_SWEEP_JITTER_WINDOW_S:-120}"
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -ge 16 && "$_JITTER_WINDOW" -gt 0 ]]; then
    _jitter=$(( (SLURM_ARRAY_TASK_ID * 7) % _JITTER_WINDOW ))
    echo "[array_task] startup jitter: sleeping ${_jitter}s (task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT)"
    sleep "$_jitter"
fi

# THE FIT POOL, read HERE, when the task starts, not at submit: a task queued across a pool
# restarted on another node then joins the new one. The address is REACHED, not just read, as the
# collector's is below: address.txt is removed by a pool that stops, but survives one that crashed.
# A task with no pool exits before it spends anything, with EX_CONFIG like a missing collector:
# every fit would fail. The probe is the one `submit` refuses a sweep on.
#
# The pool's build root becomes this run's (PETAB_MCP_BUILD_ROOT, gemot#105): one folder on shared
# storage, beside the address, for every server of the pool. Nothing here clears it; the operator
# does, with the pool stopped.
ADDRESS_FILE="$FITPOOL_DIR/address.txt"
if [[ ! -f "$ADDRESS_FILE" ]]; then
    echo "[array_task] FATAL: no fit pool address at $ADDRESS_FILE. Start it with" \
         "'python -m gemot.sweep fit-pool start' and resubmit." >&2
    exit 78   # EX_CONFIG: the environment is wrong, not the code
fi
PETAB_MCP_FIT_POOL="$(tr -d "[:space:]" < "$ADDRESS_FILE")"
if ! python -c 'import sys; from gemot.sweep.submit import probe; probe(sys.argv[1])' \
        "$PETAB_MCP_FIT_POOL" 2>/dev/null; then
    echo "[array_task] FATAL: the fit pool at $PETAB_MCP_FIT_POOL is not reachable (stale" \
         "address file, or the pool died). Restart it with 'python -m gemot.sweep fit-pool" \
         "start' and resubmit." >&2
    exit 78
fi
export PETAB_MCP_FIT_POOL
export PETAB_MCP_BUILD_ROOT="$FITPOOL_DIR/build"
echo "[array_task] fitting on the pool at $PETAB_MCP_FIT_POOL, build root $PETAB_MCP_BUILD_ROOT"

# Cores for the segfault investigation (2026-08-14: five draft-arm runs lost to the MCP server
# being killed by signal 11, cause unknown). Best-effort and cheap:
#   * `ulimit -c unlimited` — the shell default is 0, so nothing is written at all today.
#   * cores land wherever the NODE's core_pattern points. On these nodes that is
#     systemd-coredump (node-local, typically root-only), so this alone may yield nothing the
#     job can read — which is why runner/main.py also sets PYTHONFAULTHANDLER=1, dumping a
#     C-level traceback of every thread into server.log, an artifact we always archive.
#   * if the node happens to write cores to CWD, the archive step below collects them.
# Never fail the task over any of this: a missing core is worse debugging, not a broken run.
ulimit -c unlimited 2>/dev/null || echo "[array_task] could not raise the core limit (continuing)"

# DURABLE per-task working directory, not scratch. The staging exists to stop two tasks of the
# same example colliding on <example_dir>/work (see the header) -- it never needed to be
# ephemeral, and being ephemeral cost us real data: `archive_run` only runs when the harness
# RETURNS, so a task that is OOM-killed, cancelled or wall-clocked writes no archive at all, and
# the unconditional teardown below then deleted the on-disk evidence (models built so far,
# provenance.jsonl, the drafts) that was sitting right there. Measured 2026-08-16: 15 OOM-killed
# + 8 cancelled runs lost everything this way, and the post-mortem had to fall back on the trace
# collector.
#
# So: work lives under slurm/runs/work/<job>_<task> and SURVIVES a death. On the way out we drop
# only what is large and reproducible (.amici_build), and we drop the rest ONLY when the run
# archived itself successfully -- i.e. exactly the failures you would want to inspect are the
# ones kept. Override the root with PETAB_MCP_SCRATCH if a node-local disk is ever wanted.
WORK_ROOT="${PETAB_MCP_SCRATCH:-$REPO_ROOT/slurm/runs/work}"
STAGE="$WORK_ROOT/${JOB_ID}_${TASK_ID}"
rm -rf "$STAGE"

# RETENTION SWEEP, at task start. Kept work dirs are unbounded otherwise: the teardown below
# only runs when the task exits normally, and the failures it preserves (OOM SIGKILL, cancel /
# wall-clock) are exactly the ones that never reach it — so their build trees survive too and
# nothing ever collects them. Pruning here (cheap, and off the critical path of a dying task)
# bounds the shared tree by AGE; anything still wanted by then is in _runs_archive.
RETAIN_DAYS="${PETAB_MCP_WORK_RETAIN_DAYS:-7}"
# ONLY against the repo-default root, and ONLY on dirs older than the retention window.
#
# Two hazards this avoids, both learned the hard way in review:
#  * CONCURRENT TASKS SHARE THIS ROOT. An earlier version also deleted `.amici_build` from every
#    work dir regardless of age, excluding only the starting task's own — which would delete the
#    build tree out from under a sibling task mid-compile. Age alone is the safe discriminator:
#    a dir untouched for RETAIN_DAYS cannot belong to a live task.
#  * PETAB_MCP_SCRATCH IS AN OPERATOR-SUPPLIED PATH. It used to decide only where one directory
#    was created and removed; making it the root of a recursive age-based rm -rf would delete
#    unrelated content from any shared scratch someone points it at. So the sweep is skipped
#    entirely when the root is overridden.
if [[ -z "${PETAB_MCP_SCRATCH:-}" && -d "$WORK_ROOT" ]]; then
    find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$RETAIN_DAYS" \
         -exec rm -rf {} + 2>/dev/null || true
fi
mkdir -p "$STAGE/examples/$EXAMPLE"
# gemot resolves examples/ + _runs_archive/ relative to the repo it is installed from,
# which for this scratch copy is the wrong tree -- pin it to the real checkout so every task
# archives to ONE central examples/_runs_archive/ (see gemot/paths.py).
export PETAB_MCP_REPO_ROOT="$REPO_ROOT"
# Copy the example's own files (so __file__ resolves inside scratch -- see the header note),
# skipping work/ and .amici_build/ so each task starts clean and builds privately. The `*`
# glob already excludes dotfiles like .amici_build; `work` is skipped explicitly.
for entry in "$REPO_ROOT/examples/$EXAMPLE"/*; do
    name="$(basename "$entry")"
    [[ "$name" == "work" ]] && continue
    cp -r "$entry" "$STAGE/examples/$EXAMPLE/$name"
done

RUN_ARGS=(--driver "$DRIVER" --n-starts "$N_STARTS" --recursion-limit "$RECURSION_LIMIT")
if [[ -n "${MAX_WALL_SECONDS:-}" && "$MAX_WALL_SECONDS" != "None" ]]; then
    RUN_ARGS+=(--max-wall-seconds "$MAX_WALL_SECONDS")
fi
# Every LLM driver takes a model and temperature; only `scripted` calls no model. A reasoning effort
# is passed only when the sweep set one ("None": run.py's own default, which differs by driver).
if [[ "$DRIVER" != "scripted" ]]; then
    RUN_ARGS+=(--model "$MODEL" --temperature "$TEMPERATURE")
    [[ "$REASONING_EFFORT" != "None" ]] && RUN_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

# Ablation arms need NO plumbing here, and must not acquire any. Every arm is an environment
# variable (`python -m gemot.ablations`), submit --arm puts it in --export=ALL, and
# run.py reads it from the environment. Translating one into a run.py flag here would make this
# script the only place it can be set, and would keep it out of result.json.

# Phoenix tracing is MANDATORY, and with PHOENIX_ENABLED=1 a missing collector is FATAL — the
# task exits here, before it spends a single CPU-hour.
#
# There used to be a fallback: each task started its own throwaway in-process Phoenix and carried
# on. That is strictly worse than not running. On 2026-08-16 all 40 tasks of a 4-cell A/B took it
# (the endpoint was published to slurm/phoenix_data/ while the tasks, carrying an inherited
# PHOENIX_WORKING_DIR from the submitting shell, looked in .phoenix/) and every trace went to a
# per-task app that died with the task: ~26 CPU-hours of agent runs whose conversation-level
# record — the only thing that answers "what did the agent DO" — did not exist, and nobody found
# out until the pull came back 0/40. A run that cannot be analysed has no reason to have run.
#
# The endpoint is REACHED, not just read: a stale endpoint.txt outlives a dead collector (the file
# is removed by a graceful stop, not by a crash), and a task pointed at a dead endpoint loses its
# traces exactly as completely as one pointed at nothing. One GET, 10s cap, PROXIES DISABLED — the
# collector is an intra-cluster host, so an http_proxy inherited from the submitting shell would
# route the probe out to a proxy that cannot reach it and fail every task of a healthy sweep.
# Uses python (conda env is active by here) rather than curl: same probe as submit's, and it
# does not make curl's presence a new hard dependency of every run.
#
# The endpoint is the one `submit` resolved and exported, never re-derived here from an inherited
# PHOENIX_WORKING_DIR: that is the misroute above, and importing gemot sets that variable to the
# checkout's .phoenix/ in every shell that submits.
if [[ "${PHOENIX_ENABLED:-0}" == "1" ]]; then
    if [[ -z "${PHOENIX_COLLECTOR_ENDPOINT:-}" ]]; then
        echo "[array_task] FATAL: PHOENIX_ENABLED=1 but submit exported no collector endpoint." \
             "Start it with 'python -m gemot.sweep phoenix start' and resubmit." \
             "Refusing to run untraced — an untraced run cannot be analysed." >&2
        exit 78
    fi
    if ! python -c 'import sys, urllib.request as u, urllib.error as e
# build_opener with an EMPTY ProxyHandler ignores http_proxy/https_proxy entirely.
# An HTTPError (401/403/404) is the server ANSWERING -- proof of life. Only a transport
# failure means the collector is gone.
try:
    u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=10).close()
except e.HTTPError:
    pass' \
            "$PHOENIX_COLLECTOR_ENDPOINT" 2>/dev/null; then
        echo "[array_task] FATAL: Phoenix collector at $PHOENIX_COLLECTOR_ENDPOINT is not" \
             "reachable (stale endpoint file, or the collector died). Restart it with" \
             "'python -m gemot.sweep phoenix start' and resubmit. Refusing to run untraced." >&2
        exit 78
    fi
    echo "[array_task] tracing to shared Phoenix collector: $PHOENIX_COLLECTOR_ENDPOINT"
    [[ -n "${PHOENIX_PROJECT:-}" ]] && RUN_ARGS+=(--phoenix-project "$PHOENIX_PROJECT")
    # Belt and braces one layer down: if anything strips the endpoint between here and the
    # tracer setup, gemot.tracing raises instead of silently launching a local app.
    export PETAB_REQUIRE_COLLECTOR=1
fi
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
    RUN_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-petab-mcp-sweep}")
fi

RC=0
(cd "$STAGE/examples/$EXAMPLE" && python run.py "${RUN_ARGS[@]}") || RC=$?

TASK_RESULTS="$RESULTS_DIR/${JOB_ID}_${TASK_ID}"
mkdir -p "$TASK_RESULTS"
for f in result.json server.log amici_build.log run_live.json; do
    [[ -f "$STAGE/examples/$EXAMPLE/work/$f" ]] && cp "$STAGE/examples/$EXAMPLE/work/$f" "$TASK_RESULTS/"
done
# Any core the node wrote next to the server (only when core_pattern is a plain path, not the
# systemd-coredump pipe) — see the ulimit note above. Bounded and best-effort: cores are large,
# so keep at most one and never fail the task over it.
for c in "$STAGE/examples/$EXAMPLE/work"/core*; do
    [[ -f "$c" ]] && { cp "$c" "$TASK_RESULTS/" 2>/dev/null && echo "[array_task] core dumped -> $TASK_RESULTS/$(basename "$c")"; break; }
done

# NOTE: the model history is NOT copied here. `explorer.archive.archive_run` already does it on
# the completion path (runner/main.py) and copies MORE — HEAD, provenance.jsonl and every version's
# meta.json/model.ant/parameters.tsv/simulations.json — into examples/_runs_archive/<run_id>/. A
# bash copier here was a second, divergent implementation that silently dropped simulations.json,
# and its premise ("history is destroyed when a task COMPLETES") was simply false. The only real
# gap was the KEY: the archive is keyed by run id, while a sweep's results are keyed by
# <job>_<task>. result.json now carries `archive_dir` to bridge that.
{
    echo "example=$EXAMPLE"
    echo "driver=$DRIVER"
    echo "model=$MODEL"
    echo "temperature=$TEMPERATURE"
    echo "reasoning_effort=$REASONING_EFFORT"
    echo "repeat=$REPEAT"
    echo "exit_code=$RC"
    # The fit knobs as this task exported them, raw (empty = not set here), keyed as gemot's
    # result.json `fit_settings`.
    for knob in PETAB_MCP_FIT_PER_START_TIMEOUT PETAB_MCP_FIT_N_WORKERS \
                PETAB_MCP_FIT_STARTS_PER_TASK PETAB_MCP_FIT_TASK_SECONDS \
                PETAB_MCP_FIT_AMICI_ATOL PETAB_MCP_FIT_AMICI_RTOL \
                PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE \
                PETAB_MCP_CRITERION PETAB_MCP_FIT_POOL PETAB_MCP_BUILD_ROOT; do
        echo "$knob=${!knob:-}"
    done
} > "$TASK_RESULTS/combo.env"

# TEARDOWN, conditional. Whatever was built in the stage (the pool's build root is shared, and is
# the operator's to clear) is large and fully reproducible, so it always goes. The work dir itself
# is kept unless the harness archived the run (result.json carries
# `archive_dir` only on that path) -- a kept work dir means "this run died without archiving,
# its evidence is here".
rm -rf "$STAGE/examples/$EXAMPLE/.amici_build" 2>/dev/null || true
if [[ -f "$TASK_RESULTS/result.json" ]] \
   && grep -q '"archive_dir"' "$TASK_RESULTS/result.json" 2>/dev/null; then
    rm -rf "$STAGE"
else
    echo "[array_task] NOT archived (died before the harness returned) — work kept at $STAGE"
fi
echo "[array_task] task=$TASK_ID done, exit_code=$RC -> $TASK_RESULTS"
exit "$RC"
