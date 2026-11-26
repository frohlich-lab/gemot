# Running a sweep on Slurm

`python -m gemot.sweep` runs an application checkout's examples as one Slurm job array, every run
with its own server, and every server's fits on one shared pool of SLURM workers
([`src/gemot/sweep/`](../src/gemot/sweep/__init__.py)). Each mechanism is explained where it lives:
the scripts' comments and the modules' docstrings.

Nothing here runs on the login node but submissions. The environment build, every run and every
service is a Slurm job.

## What the application keeps

Nothing in the driver knows the cluster: its partitions, nodes, limits and account are the
application's, in two settings files.

- `examples/<name>/run.py`, each with its own `AGENT_N_STARTS` and `AGENT_PER_START_TIMEOUT`
  when it needs them.
- `slurm/setup_env.sh` and `slurm/environment.yml`, which build `envs/petab-mcp/` from its
  `pyproject.toml`, and with it the gemot it pins. They cannot come from the package they install.
- `slurm/sweep.args`, its `submit` flags, read before the command line:

  ```
  # The run configuration, as run.py's RUN_DEFAULTS
  --models z-ai/glm-5.3-flash --temperatures 1.0 --reasoning-effort max
  --job-name app-gemot --mem 32G
  # The cluster: the partition, nodes to keep off, the account's per-user CPU cap
  --partition compute --exclude node017,node042 --qos-cpus 2000
  ```

- `slurm/fit_pool.env`, the pool's settings, read under any given in the environment:

  ```
  FITPOOL_MAX_JOBS=12          # worker jobs while batches wait for a worker
  FITPOOL_WORKER_CORES=100     # workers per job, a core each
  FITPOOL_TIME=3-00:00:00      # past every sweep the pool serves, within the partition's limit
  FITPOOL_PARTITION=compute
  FITPOOL_EXCLUDE=node017,node042
  ```

The account, partition and QOS can also come from Slurm's own `SBATCH_ACCOUNT`,
`SBATCH_PARTITION` and `SBATCH_QOS` in the environment. Every job here inherits them, the pool's
worker jobs included, and the settings above override them.

Every command below runs from the checkout's root, with its environment active:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ./envs/petab-mcp
```

Everything the driver writes goes under the checkout's `slurm/`: `runs/` (combos files, task logs,
results, kept work dirs), `logs/` (the service jobs'), `fit_pool/`, `phoenix_data/` and
`explorer_data/`.

## 1. Start the fit pool

```bash
python -m gemot.sweep fit-pool start
```

```bash
python -m gemot.sweep fit-pool status
```

`start` submits one job running `petab-mcp-pool` (PEtab-MCP's README, "Fitting on SLURM"). Its
scheduler publishes `slurm/fit_pool/address.txt` once it listens, and it submits worker jobs while
batches wait for a worker. Every server of the pool builds into `slurm/fit_pool/build/`, which must
be on storage the compute nodes share, with locks that hold across nodes.

| Variable | Default | Notes |
| --- | --- | --- |
| `FITPOOL_MAX_JOBS` | required | worker jobs while batches wait |
| `FITPOOL_WORKER_CORES` | required | one single-threaded worker a core |
| `FITPOOL_TIME` | required | the pool's own job |
| `FITPOOL_WORKER_TIME` | `FITPOOL_TIME` | no shorter: a worker job that ends is never replaced |
| `FITPOOL_MIN_JOBS` | `0` | worker jobs kept while idle |
| `FITPOOL_WORKER_MEM` | 2 GiB a worker | per worker job |
| `FITPOOL_CPUS` / `FITPOOL_MEM` | `2` / `8G` | the pool's own job: about one core and 1.4 GB at 1200 workers and 100 servers |
| `FITPOOL_PARTITION`, `FITPOOL_EXCLUDE` | none | the pool's job and its workers |
| `FITPOOL_NODELIST` | | a node, to restart the pool at its old address |
| `FITPOOL_PORT` | `8786` | another checkout's pool on the same node needs another |

- **Restart the pool between sweeps.** Its worker jobs are not replaced when they end, so a pool
  only shrinks.
- **Restart it after a gemot pin bump.** A server refuses a pool from another environment.
- **A restart on the same node keeps the address:** `stop`, then `start` with `FITPOOL_NODELIST`
  set to the node `status` names. Tasks that start later read the address again either way. The
  runs already running survive it only once PEtab-MCP#216's last item lands: until then a restart by
  resubmission (minutes, past the client's 30 s window) ends them as
  `infrastructure_error: fit_pool`.
- **The build root is the operator's.** Clear `slurm/fit_pool/build/` only with the pool stopped.

## 2. Start the shared Phoenix collector

Tracing is mandatory. A sweep refuses to submit without a live collector unless you pass
`--no-phoenix`.

```bash
python -m gemot.sweep phoenix start
```

```bash
python -m gemot.sweep phoenix status
```

`PHOENIX_PARTITION`, `PHOENIX_CPUS` (8), `PHOENIX_MEM` (64G) and `PHOENIX_TIME` (7 days) size
its job.

## 3. Submit a sweep

A sweep is refused unless the pool's address is published and answers. Each task reads it again
when it starts and exits (EX_CONFIG, 78) if it no longer answers.

First the smoke test: one task, the scripted plumbing model, no LLM. Its log shows how long a
compile and a multistart fit of the problem take, on the pool:

```bash
python -m gemot.sweep submit --driver scripted --no-phoenix
```

Then the agent panel. Compute nodes must reach openrouter.ai, and `OPENROUTER_API_KEY` must be
exported on the login node:

```bash
python -m gemot.sweep submit --driver deepagents --repeats 5
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--examples` | every `examples/*/run.py` | |
| `--driver` | `scripted` | `deepagents`, `codex`, `claude`, `scripted`, `barebones` |
| `--models` | `glm-flash` | OpenRouter id(s) or aliases from the models file: `GEMOT_MODELS`, as an absolute path, else gemot's `models.toml` |
| `--temperatures` / `--reasoning-effort` | `1.0` / `run.py`'s | |
| `--repeats` | `1` | runs per (example, model, temperature) |
| `--n-starts` | `100` | an example's `AGENT_N_STARTS` raises it |
| `--cpus-per-task` / `--mem` / `--time` | `8` / `16G` / `08:00:00` | for the server's compiles and simulations; the fits run on the pool |
| `--partition` / `--exclude` | none | |
| `--qos-cpus` | none | the account's per-user CPU cap, which the pool and the tasks share |
| `--throttle` | what the pool leaves of `--qos-cpus`, else every task | tasks at once |
| `--arm NAME=on\|off` | | a gemot ablation arm (`python -m gemot.ablations`) |
| `--dry-run` | | write the combos file and print the `sbatch` line, submit nothing |

Under a per-user CPU cap, the pool's `FITPOOL_MAX_JOBS` × `FITPOOL_WORKER_CORES` (and its own job)
come first, then `throttle` × `--cpus-per-task`, and a larger throttle is refused. Tasks holding
the cap would keep the worker jobs pending, and every fit would wait for a worker while its task's
wall time ran out.

Each task stages a private copy of its example, so repeats cannot collide. Its `result.json`,
`server.log`, `amici_build.log` and `combo.env` land in `slurm/runs/results/<job>_<task>/`, and
the full run history in `examples/_runs_archive/`.

## 4. Stop the pool

```bash
python -m gemot.sweep fit-pool stop
```

`stop` sends SIGTERM. The pool unpublishes its address, waits (at most 10 min) until no batch is
processing, and closes its cluster, which cancels its worker jobs; `stop` waits for both, and says
so if a worker job outlives the pool. `stop --force` cancels the pool's job and its worker jobs at
once, failing every fit in flight.

## 5. Browse runs

```bash
python -m gemot.sweep explorer start
```

```bash
python -m gemot.sweep explorer status
```

`status` prints the SSH tunnel to use. Start Phoenix before the explorer.
