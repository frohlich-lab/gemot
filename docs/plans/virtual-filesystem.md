# Plan: a workspace for the panel

**Status:** a proposal for review. Nothing here is implemented.

## Goal

Give the panel more ways to work with the problem. Today the agents take in data only through MCP
reads: paged tables, SQL over the fit, and compact summaries. With a workspace they can also read
whole tables as files and run Python over them. That means their own features, statistics and
signal analysis, and simulating a draft model before committing it.

The model still changes, and is fitted, only through `make_changes`.

## Decisions

- **`PETAB_WORKSPACE` has four levels:** `off` (the default), `lead`, `analysts` and `panel` (the
  lead and the analysts). The biocurator never gets the workspace, because its blind keeps the data
  from it.
- **The tools are deepagents' own:** `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
  and `execute`. They run over gemot's existing bwrap backend (`agent/sandbox.py`), as on
  barebones, and the sandbox stays in gemot.
- **PEtab-MCP writes the copy of the problem.** That is the only server change.
- **The sandbox holds no fitting packages,** and a prompt rule forbids fitting in it. AMICI and
  scikit-learn stay.
- **The sandbox uses one core.** Barebones keeps its cores.
- **A command runs for at most 5 minutes.**
- **`biology_blind` is not supported.**

## Design

### The copy of the problem (PEtab-MCP)

- **The server takes a new flag, `--workspace DIR`.** With it, the server writes `DIR/problem/` at
  startup and wherever HEAD moves (after `Provenance.set_head`: a commit, a fork, a restore).
  - The copy is the training problem as the server holds it, as PEtab v2 files: `problem.yaml`, the
    tables, `parameters.tsv` with the fitted values, `model.xml`, `model.ant` and `inputs.tsv` if
    the problem has one.
  - A `VERSION` file names HEAD.
  - The copy is written to a fresh directory, swapped in by rename, with files at `0444`.
- **It is an allowlist by construction.** It is the server's own state, so nothing else in
  `problem/` can reach the agent:
  - not `reference.json` or `validation_v1_baseline.json`;
  - not the held-out sets, `indra/` or `run.py`;
  - nothing under `work/`.
- **The server never reads the copy back.**

### The workspace and its tools (gemot)

- **The workspace is a fresh directory per run,** outside `examples/`, with a neutral name.
  `problem/` sits in it, and the rest is scratch.
- **The backend is `sandbox_backend_class()`:** deepagents' `FilesystemBackend` in virtual mode
  plus the bwrap `execute`. gemot passes it to `create_deep_agent` as `backend=`. Four additions:
  - `--ro-bind` over `/work/problem`, and the file tools refuse writes under `/problem/`. deepagents'
    `FilesystemPermission` cannot do this with an executing backend;
  - host paths are scrubbed from file-tool errors, because the example's path names the benchmark;
  - one-core mode: CPU affinity pinned to one CPU, and the thread knobs (`OMP_NUM_THREADS`,
    `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `AMICI_PARALLEL_COMPILE`) set to 1. It is off for
    barebones;
  - the package allowlist (below).
- **Who holds the tools, per level:**

  | Level | lead | mathematician, systems_scientist, signal_engineer | biocurator |
  |---|---|---|---|
  | `off` | as today | as today | as today |
  | `lead` | ✓ | | |
  | `analysts` | | ✓ | |
  | `panel` | ✓ | ✓ | |

  - The lead's `_builtin_tool_filter` keeps the built-ins when the lead holds the workspace (it
    always strips `write_todos`) and strips them as today otherwise.
  - deepagents hands every declarative subagent the parent's backend. So a subagent without the
    workspace (the biocurator always, the analysts at `lead`) is excluded through its own harness
    profile's `excluded_tools`. The `subagents.py` note saying this leaked dates from deepagents
    0.6.12. If it still leaks on 0.7.1, that subagent is built as a `CompiledSubAgent` with no
    filesystem middleware.
- **What writing means:** a scratch file is the agent's notebook. The server is started on `work/`,
  and the judge scores the server's best version from the MCP session. Neither ever reads the
  workspace. After the run, gemot archives the scratch files beside the result.

### The sandbox

- **Packages:** one definition, in `agent/sandbox.py`.
  - `SANDBOX_PACKAGES` is the base set: amici, petab, python-libsbml, antimony, numpy, scipy,
    pandas, sympy, h5py, scikit-learn.
  - `FITTING_PACKAGES` is pypesto and fides, for barebones only.
  - Both prompts render their lists from these tuples, and the preflight imports every package.
- **What the panel's sandbox binds:** only the base set and the packages it depends on. pyPESTO,
  Fides, petab_mcp and gemot are invisible inside. It is an allowlist because hiding packages by
  name would let the next one through. Barebones keeps binding the whole environment, so that arm
  does not change.
- **Fitting is the server's.** `scipy.optimize` cannot be hidden, because `scipy.stats` imports it.
  So the prompt draws the line: do not fit the model's parameters in the workspace, with
  `scipy.optimize` or by hand, and commit through `make_changes` instead.
- **5 minutes per command, no background jobs.** The process group dies with the command. 5 minutes
  is well under the driver's 30-minute stall floor (`_STEP_STALL_TIMEOUT_S`). A single-core AMICI
  compile of the largest models may not fit in that time.
- **No route to the server.** The sandbox has its own network and PID namespaces, so it cannot call
  `evaluate_validation_data` or see the server's command line.
- **No token cap.** Nothing here touches a model call.

### Prompts

- **One constant, `WORKSPACE_TMPL`,** in a new `workspace` prompt group. It goes into the prompt of
  each agent that holds the tools. It states:
  - the layout;
  - the shell: one core, 5 minutes, no network;
  - the packages;
  - the v2 file description, shared with barebones' prompt;
  - `_WORKSPACE_RULE`: nothing here is scored or reaches the server, do not fit here, and the only
    score is the server's.
- **The lead's prompt.** When the lead holds the tools, its "You directly read only…" sentence gets
  a variant: it may read and check files, and bulk data stays with the analysts. At `off`, every
  prompt is byte-identical to today's.

### How it lands

- **`PETAB_WORKSPACE` is an ablation with levels.** `ablations.py` gains `levels`, and `arm()`
  names a deviation as `workspace=panel`.
  - gemot passes `--workspace` to the server through the bootstrap's fixed-arity arguments.
  - Any level but `off` is refused with `biology_blind`, on `--driver barebones` and on
    `--driver scripted`.
- **`off` is today's configuration.** `ablation_arm` stays `baseline`.
- **`prompt_version` moves for every run,** because it hashes all prompts. Compare by
  `prompt_group_versions()`.
- **The server pin moves too,** so the A/B is all levels at one gemot commit and one server pin.

## Checks

1. **The copy is the served problem and nothing else.** On a fixture whose problem directory also
   holds `reference.json`, `validation_v1_baseline.json`, a held-out set and a `TRUE_ANT` marker,
   the workspace holds only the copy's files (PEtab-MCP).
2. **The copy follows HEAD** after a commit, a fork and a restore, and a refused commit leaves it
   unchanged (PEtab-MCP).
3. **The sandbox's limits hold:**
   - `/problem/` refuses writes from the shell and from the file tools;
   - nothing above the workspace is visible;
   - the server's port does not connect;
   - `import pypesto` fails;
   - `nproc` prints 1;
   - a command past 5 minutes is killed with its process group.
4. **Who holds what:** per level, read off the built graph; the biocurator holds no file tool.
5. **Scoring stays the server's:** a better model in scratch does not change the judged result.
6. **Prompts:** the `off` prompts are unchanged, and the barebones prompt renders byte-identical to
   today's.
7. **The ablation:** its levels are registered and recorded, and refused where listed above.

These run in CI once frohlich-lab/gemot#37 installs bubblewrap there. A cluster smoke run at
`panel` repeats check 3 from inside the sandbox.

## Rollout

1. **PEtab-MCP: `--workspace` and the copy.**
2. **gemot:** the sandbox additions, the package tuples, the levelled ablation, the prompts and the
   tests, with the pin bumped to step 1.
3. **gemot_iclr_2026:** bump the gemot pin and add the levels to the sweep.

PEtabAgentBench does not change.

## Open questions

1. **Copy HEAD only, or every version?** Proposed: HEAD only, since `get_version` serves the rest.
2. **Is `ANALYST_MAX_TURNS = 20` enough** once an analyst can run scripts?
3. **Shared scratch or one per agent?** Proposed: shared, with a directory per agent by convention.
