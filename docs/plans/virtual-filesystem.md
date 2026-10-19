# Plan: a workspace for the panel arms

**Status:** a proposal for review, revised after two rounds. Nothing here is implemented. Comment
on any line.

The panel arms are the deepagents driver driving the petab-mcp server. This plan gives them a
virtual filesystem holding a copy of the PEtab problem, plus a sandboxed Python. This document
calls that the **workspace**.

## Decisions from review

1. **No fitting tools on the panel.** The panel's sandbox ships no optimizer: no pyPESTO, no Fides,
   no petab-mcp. Fitting stays the server's, through `make_changes`.
   - **AMICI and scikit-learn stay** in the panel's set.
   - **A prompt rule forbids fitting** in the workspace.
2. **PEtab-MCP implements the workspace.** The server writes the copy of the problem and offers the
   file and shell tools as MCP tools. The bwrap sandbox moves there too, so both arms share it.
3. **`PETAB_WORKSPACE` has levels:** `off`, `lead` (the lead only) and `panel` (the lead and the
   three analysts).
4. **One core.** Nothing in the panel's sandbox runs on more than one core. **Barebones keeps its
   cores.**
5. **No `biology_blind`.** The workspace is refused on that arm, permanently.
6. **A command runs for at most 5 minutes.**
7. **Four workspace tools,** not deepagents' seven: `workspace_read`, `workspace_write`,
   `workspace_edit` and `workspace_execute`. Listing and grepping are shell commands.

## Why

The barebones arm (`--driver barebones`) and the panel arms differ in two things at once:

- **The editing loop.** The panel changes the model only through `make_changes`. That call fits
  and versions every change, and the panel also has analysts and a finish gate. Barebones edits
  files on disk and is scored on whatever it leaves behind.
- **The tooling.** Barebones has files and a shell: deepagents' file tools and `execute` over a
  bwrap sandbox that holds the problem and a Python environment. The panel has neither, and sees
  the problem only through MCP reads.

So a barebones-versus-panel result cannot say which of the two moved it. With the workspace, the
panel gets the same files and the same sandbox, minus the optimizer: on the panel, fitting is part
of the editing loop, so the server keeps it. The comparison then isolates the editing loop. The
levels add a second comparison, the panel with and without the workspace, which measures the
tooling alone.

## The proposal in brief

- **The server owns the workspace.** gemot starts the server with `--workspace DIR` and the server
  registers four MCP tools over it: `workspace_read`, `workspace_write`, `workspace_edit` and
  `workspace_execute`. Without the flag, no workspace tool is registered.
- **`/problem/` in it is a read-only copy of the problem as the server holds it at HEAD.** The
  server writes it itself, every time HEAD moves. Nothing else from the example reaches the
  workspace.
- **The rest of the workspace is scratch.** Nothing the server scores comes from it, and the judge
  never reads it. The model the server scores still arrives only through `make_changes`.
- **The sandbox moves into PEtab-MCP** (`petab_mcp.sandbox`), from gemot's `agent/sandbox.py`.
  Barebones keeps its deepagents tools, but they run on the server's sandbox code.
- **The package sets are defined once, in `petab_mcp.sandbox`, as allowlists.** There is a base
  set for the panel, and barebones gets the base set plus the fitting set. The sandbox makes
  everything else in the environment invisible.
- **The panel's sandbox runs on one core,** pinned by CPU affinity.
- **`PETAB_WORKSPACE=off|lead|panel`** decides which agents hold the tools. The biocurator never
  does: it holds only the bio tools and `list_observables`.

## What happens to the server's constraints

The [Design notes](https://github.com/frohlich-lab/PEtab-MCP#design-notes) govern the model the
server holds. The workspace now lives in the server, so the plan adds one design note to its
README:

> **The workspace is not a route into the model.** Its files are scratch and a read-only copy of
> the problem. Nothing in it is read back into the problem, fitted or scored, and the sandbox
> ships no optimizer: fitting stays `make_changes`'s.

What the agent can now do outside the MCP loop:

- read the whole problem as files, the measurement table included;
- run Python against it: simulate a draft model with AMICI at given parameter values, compute
  residuals, statistics or signal features;
- draft Antimony or SBML in scratch files and check that it loads and simulates, before committing
  it through `make_changes`.

What it cannot do there: fit with a shipped optimizer, use more than one core, reach the server or
the network, or see anything but the workspace.

Design note by design note:

| Design note | On the workspace arm |
|---|---|
| No free-form editing | Holds for the model. The workspace tools write scratch files only, `/problem/` is read-only, and the server never reads a workspace file back into the problem. |
| A change and its score never come apart | Holds. No optimizer is shipped, and the prompt forbids fitting in the workspace. The agent can still simulate. A hand-written optimizer (`scipy.optimize`) is against that rule, and nothing would score its numbers anyway. |
| SBML and PEtab stay consistent automatically | Holds. The copy is the server's reconciled state. A draft the agent edits locally is not reconciled until it is committed; then the server reconciles it like any `make_changes`. |
| Every operation is versioned and logged | Holds, and now covers the workspace: the server appends every workspace call to its provenance directory (`workspace.jsonl`). Scratch files are not versioned; gemot archives them with the run. |
| Iteration is cheap; results are not cached | Unchanged. The server still refuses revisits. |
| Comparisons must be valid; one score | Holds. The server's criterion is the only score. |
| The estimation scale is the server's | Holds. `/problem/parameters.tsv` carries the server's `parameterScale` column, for reading. |
| A hardcoded constant is not free | Unchanged. The server charges literals on commit. |

## 1. The copy of the problem

### What it holds

It holds exactly the training problem the server serves, as PEtab v2 files at HEAD:

- `problem.yaml` (v2), `measurements.tsv`, `conditions.tsv`, `experiments.tsv`, `observables.tsv`;
- `parameters.tsv`, the server's parameter table at HEAD: the shipped rows, the rows reconcile
  added, the fitted `nominalValue`s and the `parameterScale` extension column;
- `model.xml` (SBML) and `model.ant` (the canonical Antimony) of HEAD, or the shipped model before
  the first commit (empty on a zero-shot problem);
- `inputs.tsv` when the problem has one (fujita). The server already reads it and shows its rule
  to the agent;
- `VERSION`, with the server's HEAD number and label, so a reader knows which version the files
  are.

The agent can already read all of this through MCP: `get_measurements`, `list_conditions`,
`list_experiments`, `list_observables`, `get_parameters` and `get_model`. The copy adds a format,
not information.

A v1 problem appears as the v2 problem the server upgraded it to, as `list_experiments` already
shows it. So the copy is always v2, and one prompt describes it.

### Per scenario

| Scenario | The server loads | The copy | Kept out |
|---|---|---|---|
| Benchmark example (`paths.BENCHMARK_EXAMPLES`), materialized as v2 | `work/`, a copy of `problem/` | the files above | `problem/reference.json`; `problem/validation_v1_baseline.json` (perelson, crauste, fujita, lucarelli, raimundez); `validation/`; `validation_source/`; `indra/` (`entity_map.json`, `statements.json`, `entity_ids.json`); `run.py` (`TRUE_ANT`); `.amici_build/` |
| Data package (`paths.DATA_PACKAGE_EXAMPLES`), committed v2, reference-free | the same | the files above | `validation/` (the committed held-out set) |
| Synthetic demo (`paths.SYNTHETIC_EXAMPLES`) | the same, upgraded to v2 by the server if v1 | the server's v2 view | `problem/reference.json`, `run.py`, the data generator |
| `indra_statements` arm | the same, plus the empty entity map in the build root | the files above; the model is whatever the statements built | the problem's `indra/` (the reference's agents); the empty map stays on the harness side |
| `biology_blind` arm | the anonymized `work/` | none: the workspace is refused on this arm | everything |

Also never in the workspace:

- anything under `work/`. `run_live.json` there carries the reference score, and the directory
  also holds `result.json`, `server.log` and the version store `.petab_mcp/`;
- the private blinding directory, the build root and `$HOME`;
- the harness and server checkouts.

### How it is written

**The server writes the copy from its own state.** It is therefore an allowlist by construction:
the problem the server serves never contains an answer key, and the held-out set it holds for
`evaluate_validation_data` stays in memory and out of the copy. A file added to `problem/` later
does not reach the agent by default. A list of withheld names, like barebones' `_WITHHELD`, cannot
promise that (see "Found while reading").

- **It is written with the writers the version store already uses** for `versions/vN/`.
- **It goes into a fresh directory, swapped in by rename.** A command binds `/problem` when it
  starts, so it sees one version throughout. The previous copy is kept until the next swap, so a
  running command never loses its files.
- **Modes are `0444` for the files and `0555` for the directory.**

### Following the server

The copy follows HEAD, and it is written where HEAD moves:

- `Provenance.set_head` is the one writer of HEAD. The session writes the copy right after it,
  from the state it has just applied: a commit, a `base_version` fork, `apply_version`
  (`restore_version` and `rollback_to`).
- At startup, the server writes the shipped problem as the first copy.
- The copy is written under `session.lock`, which `make_changes` already holds. A workspace call
  never takes the lock, so a long command does not delay a commit.

A failed write leaves the previous copy, with its own `VERSION`, in place. It is logged and does
not fail the commit: the copy is a view, and the version store remains the record.

Only HEAD is copied, not every version: earlier versions are one `get_version` call away. See open
question 1.

## 2. Read-only and writable

On the host, and as a command sees it:

```
host                                   inside the sandbox
<tmp>/gemot-ws-<uuid>/                 /work         (cwd, read-write)
├── problem/   0555, swapped per HEAD  /work/problem (read-only)
├── .home/  .tmp/                      $HOME, /tmp
└── anything the agent writes          /work/...     (read-write)
```

**The four tools all run inside the sandbox.** `workspace_read`, `workspace_write` and
`workspace_edit` are small sandboxed commands too; content is passed on stdin. So one mechanism
confines all four, and the read-only rule is the mount's:

- the paths an agent gives are `/work` paths, and nothing outside `/work` exists to name;
- `/problem/` is `--ro-bind`, so a write there fails with EROFS whichever tool tries it. The
  tool's message says the copy is the server's and changes go through `make_changes`;
- this costs one bwrap start per call, which is milliseconds, against a fit's minutes;
- the host files are also `0444`.

**The tool names are distinct from deepagents' built-ins.** deepagents gives every agent its own
`read_file`, `write_file`, `edit_file` and `execute`. The lead's `_builtin_tool_filter` strips those
by name, and the analysts keep them over an empty in-state filesystem, because they cannot be
filtered (`subagents.py`). A server tool named `execute` would be stripped from the lead along
with them. The `workspace_*` prefix keeps the two apart.

**There are four tools, not deepagents' seven** (decided in review). Barebones holds `ls`,
`read_file`, `write_file`, `edit_file`, `glob`, `grep` and `execute`. On the panel, listing,
globbing and grepping are shell commands through `workspace_execute` (`ls`, `find`, `grep`). Every
tool's schema is re-sent on every turn, the cost `tools.py` curates against, so three fewer tools
is a smaller prompt for every agent that holds them. The panel matches barebones in what it can do,
not tool for tool.

**The workspace root has a neutral name.**

- gemot creates it in a temporary directory outside `examples/`, with a name that says nothing
  about the example, and passes it to the server.
- The server refuses a workspace inside or overlapping its problem directory or provenance
  directory.
- Tool output is scrubbed of the host root. The example's directory carries the benchmark's name,
  which PEtabAgentBench's `ANON_MODEL_ID` exists to keep from the agent.

**Everything outside `/problem/` is scratch,** shared by the agents that hold the tools.

**What writing means.** A scratch file is the agent's notebook: nothing reads it except the agent
and, after the run, the archive. In particular:

- the server never reads a workspace file into its problem. The workspace module has no path back
  into the session's state, and a test asserts the problem is unchanged after any workspace call;
- the judge (`runner.judge._judge`) takes the MCP caller and `work/`, and scores the server's best
  version. It is never handed the workspace path;
- a model the agent wants scored must be submitted as text to `make_changes`, which fits, versions
  and records its justification like any other.

After the run, gemot archives the scratch files beside the result in `<archive>/workspace/`. The
size cap and suffix list of `runner.barebones._archive_delivery` apply here too. `problem/` is not
archived, since the version store already holds every version. `workspace.jsonl` arrives with the
version store, which the archive already copies.

## 3. Code execution

### The sandbox moves into PEtab-MCP

gemot's `agent/sandbox.py` moves, minus its deepagents backend, to `petab_mcp.sandbox`. The moved
part is the bwrap command line, the built environment, the process-group timeout kill, the
preflight and the shell audit.

- **The server uses it** for the four workspace tools.
- **Barebones uses it too.** The deepagents backend class (`FilesystemBackend` plus `execute`)
  stays in gemot, as a thin shim whose `execute` calls `petab_mcp.sandbox`.
- **What it gives a command is unchanged:** the workspace read-write at `/work`, the system trees
  read-only, no network, no `$HOME`, an allowlisted environment and its own process group. The one
  change is that the environment prefix is no longer bound whole (next subsection).
- **The preflight runs when the server starts** with `--workspace`. On a node where it fails, the
  server exits before listening, and gemot writes a degraded result (`sandbox: ...`). gemot can run
  the same preflight first for a cleaner reason string. There is no unsandboxed fallback.

### Two package sets, one definition

Today three places define the set, and they can drift:

- the environment itself: `sandbox.default_env_prefix()`, bound whole, so every installed package
  is importable (petab_mcp, gemot and langchain included);
- the prose list in `prompts.BAREBONES_SYSTEM_PROMPT_TMPL`;
- the dependencies in `pyproject.toml`, which name pyPESTO as "promised to the barebones agent's
  sandbox". frohlich-lab/gemot#37 adds scikit-learn there; h5py is still missing.

The proposal:

- **Two tuples in `petab_mcp.sandbox`:**
  - `SANDBOX_PACKAGES`, the base set: amici, petab, python-libsbml, antimony, numpy, scipy, pandas,
    sympy, h5py, scikit-learn;
  - `FITTING_PACKAGES`: pypesto, fides.
  - The panel gets the base set. Barebones gets the base set plus `FITTING_PACKAGES`, which is
    today's list.
- **The sandbox binds only the allowlist.**
  - The interpreter and its standard library are bound read-only, as now. A tmpfs replaces
    `site-packages`.
  - Into it the sandbox binds each allowed distribution, plus its dependency closure from
    `importlib.metadata`, read-only.
  - Everything else installed in the environment is invisible inside: pyPESTO, Fides, petab_mcp,
    gemot, langchain, jax and optax.
  - This is the allowlist the copy of the problem already is. The alternative, masking named
    packages, is a list of withheld names again.
- **A test asserts the base set's closure contains no fitting package,** so a new dependency of
  AMICI or petab cannot bring an optimizer in unnoticed.
- **The preflight imports every package of the arm's set inside the sandbox.** A missing transitive
  dependency then fails the node before the run, not halfway through it.
- **The prompts render the lists from the tuples.** The server's `workspace_execute` docstring
  states the base set. gemot keeps that sentence beside its own description of the tool, as it
  keeps the server's `statements` paragraph on the indra arm (`tools._statements_route`), because
  the list is the server's. Barebones' prompt fills a `{packages}` slot from both tuples. Its
  rendered text stays byte-identical, and a test compares it with today's.
- **PEtab-MCP's `pyproject.toml` names each package as a direct dependency,** with a why-comment,
  per the style guide's "A new import means a new pin".

**Where the line on fitting sits:**

- **No fitting package is shipped.**
- **AMICI and scikit-learn stay.** AMICI simulates a draft at given values. scikit-learn's
  estimators are for statistics on the data and the residuals, not for the model's parameters.
- **`scipy.optimize` stays too.** It is part of scipy, and `scipy.stats` imports it, so it cannot
  be hidden without breaking statistics. An agent could also write a fitter by hand in numpy.
- **So the prompt draws the rest of the line.** `_WORKSPACE_RULE` (section 4) forbids fitting the
  model's parameters in the workspace, whether with `scipy.optimize` or by hand. To fit a model,
  the agent commits it with `make_changes`, which fits it.
- **The single core and the 5-minute command bound** make a local fit slow besides.
- **A fit is visible after the fact.** Every command is in `workspace.jsonl`, so the analysis can
  count the commands that call an optimizer.

### One core

The panel's sandbox runs every command on one core:

- **CPU affinity is set to a single CPU of the job's allocation** on the bwrap process
  (`os.sched_setaffinity`, before `exec`), and every process inside inherits it. Inside, `nproc`
  says 1.
- **The thread knobs are set to 1:** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`
  and `AMICI_PARALLEL_COMPILE`.
- **`PETAB_MCP_FIT_N_WORKERS` is not passed in.**

So the server's fits keep their `FIT_N_WORKERS` cores and their budgets in `fit_budget.py`.
Compiling a large model for a local simulation is slower on one core; that cost falls on the
agent's wall-clock budget, not on the server.

**Barebones keeps its cores.** Its fits run in its sandbox, and the server is not running while it
works. So the one-core mode is a parameter of `petab_mcp.sandbox`, on for the server's workspace
and off for barebones.

### Timeouts

**A workspace command runs for at most 5 minutes.** That is the server's
`PETAB_MCP_WORKSPACE_TIMEOUT`, default 300 s, a knob like the server's other `PETAB_MCP_*`
settings.

- **When a command times out,** the server kills the command's process group and returns its
  partial output with the timeout named, as barebones' `execute` does.
- **No background jobs.** Nothing outlives its call: the process group goes when the command
  returns. The prompt offers no `nohup`, unlike barebones', whose commands may run for a whole fit
  budget.
- **Some local simulations will not fit in 5 minutes.** A single-core AMICI compile of the largest
  models (alkan, bachmann) may take longer than that. Such a model cannot be simulated locally, and
  the server's fits are where it gets simulated. The smoke run measures the compile times.
- **The timeout sits well below gemot's read timeout for the call.** A call without `n_starts` gets
  the run default's fit budget (`clock._wrap_call_timeout`), so a timed-out command returns cleanly
  before the client gives up. A test pins that order.
- **The driver's stall window already covers such a call,** because its floor is that same budget.

### What the shell cannot reach

The sandbox runs next to a live server:

- **No route to the server.** `--unshare-all` gives the sandbox its own network namespace, so
  `127.0.0.1:<port>` inside is not the server. A command cannot call `evaluate_validation_data`,
  `restore_version` or any tool outside the agent's curated list.
- **No view of the server process.** The sandbox has its own PID namespace, so `/proc` shows no
  server. The server's command line names the held-out table's path.

### No token limit

Nothing here touches a model call. No `max_tokens` and no cap of any kind is added to any call,
retry or hand-off. The sandbox's existing 100 kB bound on one command's output stays, as a bound
on tool output.

## 4. Agents, tools and prompts

### Who holds what, by level

| Level | Server | lead | mathematician, systems_scientist, signal_engineer | biocurator |
|---|---|---|---|---|
| `off` | started without `--workspace`; no workspace tool registered | as today | as today | as today |
| `lead` | `--workspace` | today's tools plus the four `workspace_*` tools | as today | as today |
| `panel` | `--workspace` | today's tools plus the four | each persona's tools plus the four | as today |

- **One tuple, `WORKSPACE_TOOLS`, in `prompts.py`,** names the four tools. It is spliced into
  `ORCHESTRATOR_TOOLS` and, at `panel`, into the three analyst lists, then stripped per level by
  `tools._ARM_TOOLS`, as `get_skill` and `list_mechanisms` are today. The lead prompt renders its
  tool sentence from the same tuple (the `LEAD_SELF_READ_TOOLS` pattern), so the prompt cannot name
  a tool the agent does not hold.
- **The biocurator is excluded structurally.** It holds only the bio tools and `list_observables`,
  so it never sees a workspace tool, and its blind keeps excluding the data. No subagent rebuilding
  is needed: the tools are MCP tools, curated per persona like any other.
- **deepagents' built-ins are untouched:** stripped from the lead, kept over an empty in-state
  filesystem on the analysts.
- **`TOOL_DESCRIPTIONS` gains the four.** `workspace_read` carries the MCP read-only annotation;
  the other three do not.
- **The circuit breaker wraps the four like any MCP tool.** A non-zero exit is a result, not an
  error, and stays out of `get_errors` (`_Ctx.run(..., record=False)`).

### Prompts

New text goes into a new `PROMPT_GROUPS` group, `workspace`. A change to it is then attributable,
and the other groups' versions do not move.

- **`WORKSPACE_TMPL`** is one constant. It is composed into the lead's system prompt
  (`DEFAULT_SYSTEM_PROMPT`, or `INDRA_SYSTEM_PROMPT` on that arm) and, at `panel`, into each
  analyst's persona, with the agent's name filled in. It states:
  - the layout: `/problem/` is the server's problem at HEAD, read-only, and `VERSION` says which
    version it is. The rest is scratch, shared with the other agents that hold the tools, and each
    agent writes under a directory named for itself;
  - the shell: one core, no network, 5 minutes per command with no background jobs, and the
    packages (from the server's sentence);
  - the v2 file description, as a constant shared with `BAREBONES_PETAB_FORMATS["v2"]` and lifted
    out of it without changing its text;
  - **the rule, as one constant, `_WORKSPACE_RULE`:**
    - nothing written here reaches the server or the judge;
    - do not fit the model's parameters here, with `scipy.optimize` or by hand: the model changes
      and is fitted only through `make_changes`;
    - the only score is the server's.
- **The lead's "You directly read only…" sentence becomes false at `lead` and `panel`.** It says
  the lead reads "only the model, its SCORES … for the raw DATA … you DELEGATE to the analysts".
  At those levels the lead prompt composes a variant instead:
  - the lead may draft and check model text in the workspace;
  - bulk reading of the data stays with the analysts, for the same token reason as today;
  - the `off` text stays byte-identical.
- **`_REPORT_CONTRACT` stays true.** Its "You cannot edit the model" still holds. At `panel` the
  workspace block adds that an analyst can write and run scripts.
- **Unchanged:** the task, the biocurator, the finish gate and the nudges.

## 5. How it lands

### A levelled ablation

`PETAB_WORKSPACE` takes `off` (the default), `lead` or `panel`, and `changes_prompt=True`.

- **It meets `ablations.py`'s membership test**: it changes "what tools it holds".
- **It follows the one-mechanism rule**: every arm is an environment variable.
- **The registry needs levels.** Every ablation today is on/off. `Ablation` gains optional
  `levels`, and `ablations.level(name)` reads one.
  - `state()` records the level, and `arm()` names a deviation as `workspace=lead`.
  - `describe()` prints the levels, and `test_ablations.py` executes each printed setting, as it
    does for every switch now.
  - An unknown level is refused, as an unknown name is.
- **gemot passes the server `--workspace DIR` at `lead` and `panel`.** The bootstrap gains one
  fixed-arity argument (`runner/server.py`), `""` when off, as for `<entity_map>`.

It combines with `biocurator`, `reference_handoff`, `bio_literature` and `indra_statements`. Any
level but `off` is refused at startup in three cases:

- **with `biology_blind`.** It is not supported;
- **on `--driver barebones`**, which has its own deepagents workspace;
- **on `--driver scripted`**, which has no agent.

So a recorded `workspace=lead` or `workspace=panel` always means a panel holding the tools.

### Comparability with earlier sweeps

- **`off` is the old configuration.** It has the same tools, the same prompts and the same server
  behaviour. `ablation_arm` stays `baseline`, because `arm()` names only deviations.
- **`prompt_version` moves for every run.** It hashes every registered prompt, used or not. Compare
  across the change by `prompt_group_versions()`: `lead`, `analysts` and the other groups must not
  move.
  - The `barebones` group moves once, because its template gains the `{packages}` slot.
  - The text the barebones agent reads does not move; a test renders it and compares.
- **`state_key()` moves,** as for any appended switch. This is the trade its docstring documents.
- **Barebones' sandbox narrows.** With the allowlist, petab_mcp, gemot and langchain stop being
  importable on barebones too. That is what its prompt already says ("no server, no helper
  library"). An earlier barebones run that imported them shows it in `workspace_audit.commands`.
- **The server pin moves,** and brings whatever else is on PEtab-MCP main by then. So earlier
  sweeps are not the control. The A/B is four arms at one gemot commit and one server pin:
  `off`, `lead`, `panel` and barebones. That is the rule the retired-ablations entry in
  `docs/RATIONALE.md` states ("a fresh A/B with both arms on one code version").
- **`Scenario.version` does not move,** because no example changes.

## 6. Checks

The checks live in PEtab-MCP (`tests/test_workspace.py`, `tests/test_sandbox.py`) and in gemot
(`tests/test_workspace_arm.py`, plus `test_ablations.py`, `test_prompt_tracking.py` and
`test_barebones.py`). Each names the leak or regression it locks out.

**Isolation and leaks** (PEtab-MCP, unless noted):

1. **The copy is the served problem and nothing else.** The fixture problem directory has
   `reference.json`, `validation_v1_baseline.json` and a stray file, the server is started with
   `validation_measurements`, and a `run.py` carrying a `TRUE_ANT` marker sits one level up. The
   workspace must hold only the copy's files and the sandbox's own directories. No file in it may
   contain a held-out row, the reference `nll` or the marker.
2. **The sandbox sees nothing above the workspace:** not the problem directory, the provenance
   directory, the build root or `$HOME`.
3. **The sandbox cannot reach the server.** A connect to the server's port from inside fails, and
   `/proc` lists no server.
4. **No fitting package inside:** `import pypesto`, `import fides` and `import petab_mcp` fail in
   the panel's sandbox. Every `SANDBOX_PACKAGES` entry imports, and the base set's closure contains
   no fitting package.
5. **One core, five minutes:** inside, `nproc` prints 1 and `os.sched_getaffinity(0)` has one
   CPU. A command that sleeps past the timeout is killed with its process group, and a
   backgrounded process does not outlive its call. On barebones, the one-core mode is off.
6. **`/problem/` refuses writes** from `workspace_write`, `workspace_edit` and a shell command
   (EROFS), and a path outside `/work` cannot be named.
7. **No host path in any tool output.** The workspace root's name does not appear.
8. **No problem tree in the environment.** The preflight refuses a non-editable PEtabAgentBench
   install, because its wheel ships every problem's `run.py` and `reference.json` under
   `petab_agent_bench/problems/` (`hatch_build.py`). gemot_iclr_2026's README installs it
   editable, so this guards a future install, not today's. (gemot)

**Behaviour:**

9. **The copy follows HEAD.** After a commit, a `base_version` fork and a `restore_version`,
   `/problem/model.ant` equals `get_model()`, `parameters.tsv` equals the PEtab columns of
   `get_parameters()`, and `VERSION` is HEAD. A refused commit leaves all three untouched.
10. **The workspace is not a route into the model.** After writing a model file and `problem.yaml`
    into scratch, and attempting the same in `/problem/`, the server's problem, HEAD and version
    list are unchanged.
11. **Scoring stays the server's.** A run whose scratch holds a better model than the server's best
    is judged on the server's best. (gemot)
12. **Who holds what, per level.** At `lead`, only the lead holds the four tools; at `panel`, the
    lead and the three analysts; the biocurator never; at `off`, the server registers none. Read
    off the built graph, in a subprocess, because the arm is resolved at import time (style guide).
    (gemot)
13. **Timeout order.** The server's command timeout sits below the read timeout gemot applies to
    the call. (gemot)
14. **Prompt tracking.** The `workspace` group is registered, the `off` lead and analyst prompts are
    unchanged, and the barebones prompt renders byte-identical to today's. (gemot)
15. **The ablation's levels** are registered, recorded and refused when unknown, and any level but
    `off` is refused with `biology_blind` and on the barebones and scripted drivers. (gemot)
16. **`result.json`** carries the sandbox preflight record and the workspace call counts (from
    `workspace.jsonl`), and the archive holds the scratch files. (gemot)

**CI.** frohlich-lab/gemot#37 installs bubblewrap in gemot's CI, lifts the runner's AppArmor limit
on user namespaces, and makes a missing bwrap fail under CI instead of skipping. PEtab-MCP's CI
needs the same before the sandbox moves there.

**Cluster smoke run.** Before any sweep, run the toy problem and then one benchmark at `panel`.

- Each runs a probe inside the sandbox: `find / -name reference.json`, `import pypesto`, `nproc`,
  and a connect to the server's port.
- It also times a single-core AMICI compile of the largest model, alkan, against the 5-minute
  bound.

## 7. Rollout

| Step | Repo | What | Pins |
|---|---|---|---|
| 1 | gemot | CI runs the bwrap tests: frohlich-lab/gemot#37 | none |
| 2 | PEtab-MCP | `petab_mcp.sandbox`: the bwrap code moved from gemot, the allowlist binds, the two package tuples, one-core mode, the preflight's imports; CI with bubblewrap; the packages as dependencies | none |
| 3 | gemot | barebones on `petab_mcp.sandbox`; `agent/sandbox.py` reduced to the deepagents shim; the `{packages}` slot | `petab-mcp` to step 2 |
| 4 | PEtab-MCP | `petab_mcp.workspace`: `--workspace`, the copy written at every HEAD move, the four tools, `workspace.jsonl`, the README design note | none |
| 5 | gemot | the levelled `PETAB_WORKSPACE`: the registry's levels, the bootstrap argument, `WORKSPACE_TOOLS` in the tool lists, the `workspace` prompt group, the archive, the refusals, the tests | `petab-mcp` to step 4 |
| 6 | none | cluster smoke run with the in-sandbox probe | none |
| 7 | gemot_iclr_2026 | bump the gemot pin, add the levels to `slurm/make_sweep.py`, run the four-arm A/B | `gemot` to step 5 |

Steps 2 and 3 change no panel run and narrow barebones' sandbox to what its prompt already
promises. They can land and be checked on barebones before the workspace exists.

**PEtabAgentBench needs no change.** Its problems already keep the held-out sets and answer keys
out of what the server loads. Its `gemot` extra pins gemot only for materialization and refits, and
can follow whenever.

Step 5 adds at most one `docs/RATIONALE.md` entry. It records why the copy and the package sets are
allowlists rather than lists of withheld names. The barebones leak below is the failure that
motivates it.

## 8. Open questions for reviewers

1. **HEAD only, or every version** (`/versions/vN/`) and the HEAD fit's simulation table? Proposed:
   HEAD only, since `get_version` and `query_data` already serve the rest.
2. **Analyst turn limit:** is `ANALYST_MAX_TURNS = 20` still enough at `panel`, once an analyst can
   write and run a script?
3. **Shared scratch:** per-agent directories by convention, as proposed, or a workspace per agent?
4. **Other clients of the server.** The workspace is now a server feature, so any MCP client could
   start it, petabgpt included. Is "off unless `--workspace`" enough, or should the server also
   refuse the workspace without a confirmed sandbox (it already refuses on a failed preflight)?

## Found while reading

These are separate fixes, not part of this plan.

- **Barebones stages an answer key for five problems.**
  - `runner.barebones.stage_workspace` withholds files by name:
    `_WITHHELD = ("reference.json", "model.ant")`.
  - `problem/validation_v1_baseline.json` is tracked in `problem/` for perelson, crauste, fujita,
    lucarelli and raimundez (PEtabAgentBench `.gitignore`), so it reaches the barebones agent's
    `work/agent/`.
  - The file holds more than the reference's held-out score. It also holds the reference model's
    fitted parameters by name: `reaction_1_k1` on fujita, `geneA_act1` on lucarelli, the per-patient
    `c` and `delta` on perelson. The names give away the reference's structure.
  - frohlich-lab/gemot#38 replaces the list with an allowlist built from `problem.yaml`, plus
    `inputs.tsv`.
  - Whether earlier barebones runs on those five problems read the file can be checked in their
    `workspace_audit.commands`.
- **CI skipped every bubblewrap test,** because the runner had no bubblewrap. frohlich-lab/gemot#37
  fixes this, and is step 1 above.
- **Barebones' workspace path names the example** (`examples/<name>/work/agent`). deepagents'
  path-escape error carries that path. Not checked: whether the tool layer passes that message on
  to the agent.

## Not in this plan

- `write_todos` and deepagents' `general-purpose` subagent. Barebones has both and the panel does
  not, but neither is the filesystem or the package set.
- Any change to the editing loop itself, the finish gate or the judge.
- `biology_blind` support.
