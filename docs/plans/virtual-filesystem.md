# Plan: a workspace for the panel arms

**Status:** a proposal for review. Nothing here is implemented. Comment on any line.

The panel arms are the deepagents driver driving the petab-mcp server. This plan gives them a
virtual filesystem holding a copy of the PEtab problem, plus the barebones arm's sandboxed Python.
This document calls that the **workspace**.

## Why

The barebones arm (`--driver barebones`) and the panel arms differ in two things at once:

- **The editing loop.** The panel changes the model only through `make_changes`. That call fits
  and versions every change, and the panel also has analysts and a finish gate. Barebones edits
  files on disk and is scored on whatever it leaves behind.
- **The tooling.** Barebones has files and a shell. It gets deepagents' file tools and `execute`
  over a bwrap sandbox that holds the problem and a Python environment with AMICI, pyPESTO and
  petab. The panel has neither, and sees the problem only through MCP reads.

So a barebones-versus-panel result cannot say which of the two moved it. This plan gives the panel
the barebones tooling, so that comparison isolates the editing loop. It also adds a second
comparison: the panel with and without the workspace, which measures the tooling alone.

## The proposal in brief

- **A new ablation**, `workspace` (`PETAB_WORKSPACE=1`, default off), on the deepagents driver.
- **The workspace is a fresh directory per run**, outside `examples/`. The sandbox shows it at
  `/work` and the file tools at `/`.
- **`/problem/` is a read-only copy of the PEtab problem as the server holds it at HEAD.** It
  contains the PEtab v2 files the server serves and is re-exported after every `make_changes`.
  Nothing else from the example reaches the workspace.
- **The rest of the workspace is scratch.** Nothing written there reaches the server or the judge.
  The model the server scores still arrives only through `make_changes`.
- **`execute` is the barebones sandbox**: `sandbox.py`'s bwrap backend, with two small additions.
- **The package set becomes one constant.** Both arms' prompts render it and the sandbox preflight
  imports it.
- **The lead and the three analysts share the workspace.** The biocurator does not get it, because
  its blind excludes the data and the copy contains the data.
- **The server exports the copy** through a new read-only MCP resource. That is a PEtab-MCP change
  and a pin bump. A gemot-only fallback is open question 2.

## What happens to the server's constraints

The [Design notes](https://github.com/frohlich-lab/PEtab-MCP#design-notes) govern the model the
server holds. They still hold, because the workspace has no path into the server. The server and
the judge read nothing from it, and the sandbox cannot reach the server's port (section 3).

What the agent can now do outside the MCP loop:

- read the whole problem as files, the measurement table included;
- run Python against it: simulate a draft model with AMICI, fit it with pyPESTO, compute residuals
  or statistics;
- draft Antimony or SBML in scratch files and test it before committing it through
  `make_changes`.

Design note by design note:

| Design note | On the workspace arm |
|---|---|
| No free-form editing | Holds. `/problem/` is read-only, and no workspace file is a route into the server. The export resource is read-only and is on no agent's tool list. |
| A change and its score never come apart | Holds for the server's model. A local fit in the sandbox is unversioned exploration, like the agent's reasoning, and nothing scores it. |
| SBML and PEtab stay consistent automatically | Holds. The copy is the server's reconciled state. A draft the agent edits locally is not reconciled until it is committed; then the server reconciles it like any `make_changes`. |
| Every operation is versioned and logged | Holds for the server. Workspace activity is not versioned. Every command goes into the shell audit, and the scratch files are archived with the run (section 2). |
| Iteration is cheap; results are not cached | The server still refuses revisits. The agent can now keep results of its own, and nothing on the server depends on them. |
| Comparisons must be valid; one score | The server's criterion is the only score. A BIC the agent computes locally is not what is judged and can differ, since the server's `k` also counts numeric literals. The prompt says so. |
| The estimation scale is the server's | `/problem/parameters.tsv` carries the server's `parameterScale` column. The prompt tells the agent to pass it to pyPESTO, in the sentence barebones already uses. |
| A hardcoded constant is not free | Unchanged. The server charges literals on commit, whatever the agent did locally. |

One thing does weaken: **the version history is no longer the whole search.** An agent can try
many structures locally and commit only the best. The shell audit (every command, recorded in
`result.json`) and the archived scratch files record that half. See open question 4.

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
| `biology_blind` arm | the anonymized `work/` | refused in the first version (section 5) | `.work_biology_blinding/` (the identifier map and the held-out set) |

Also never in the workspace:

- anything under `work/`. `run_live.json` there carries the reference score, and the directory
  also holds `result.json`, `server.log` and the version store `.petab_mcp/`;
- the private blinding directory, the build root and `$HOME`;
- the harness and server checkouts.

### How it is materialized

The copy is **an allowlist by construction**. It is what the server serves, and the problem the
server serves never contains an answer key or the held-out set. A file added to `problem/` later
does not reach the agent by default. A list of withheld names, like barebones' `_WITHHELD`, cannot
promise that (see "Found while reading").

**Proposed PEtab-MCP change:** a read-only MCP resource, for example `petab://problem/files`.

- It returns the training problem at HEAD as `{"version": N, "label": ..., "files": {name: text}}`,
  written with the writers the version store already uses.
- It is a resource, not a tool: `langchain_mcp_adapters.load_mcp_tools` loads tools only, so no
  agent can be handed it by accident.
- Its contract: never a held-out row, and the tables `create_server` was given, upgraded and
  reconciled as the server holds them.

gemot reads the resource through the harness's own MCP session, never an agent's, and writes the
files:

- into a fresh directory, swapped in by rename. A command binds `/problem` when it starts, so it
  sees one version throughout. The previous copy is kept until the next export, so a running
  command never loses its files;
- with modes `0444` for the files and `0555` for the directory.

### Following the server

The copy follows HEAD:

- **It is exported once before the first agent turn**, after the preflight has scored the
  baseline.
- **It is re-exported after every `make_changes` call returns.** A wrapper on that tool object does
  it, installed where `_wrap_tools_with_circuit_breaker` wraps the tools. A call that committed
  nothing leaves HEAD where it was, and the wrapper skips the export when `VERSION` already names
  that HEAD.
- **No other tool needs a trigger.** `make_changes` is the only lead tool that moves HEAD;
  `restore_version` belongs to the judge, not the lead.

A failed export leaves the previous copy, with its own `VERSION`, in place. It is counted in
`result.json` (`workspace_mirror_failures`) and does not stop the run.

Only HEAD is copied, not every version: earlier versions are one `get_version` call away. See open
question 3.

## 2. Read-only and writable

On the host, and as the agent sees it:

```
host                                   sandbox (execute)          file tools
<tmp>/gemot-ws-<uuid>/                 /work   (cwd, read-write)  /
├── problem/   0555, swapped per HEAD  /work/problem  (read-only) /problem/  (refuses writes)
├── .home/  .tmp/                      $HOME, /tmp
└── anything the agent writes          /work/...  (read-write)    /...
```

- **The workspace root has a neutral name.** It is a temporary directory outside `examples/` whose
  name says nothing about the example.
  - The sandbox only ever shows `/work`. But deepagents' `FilesystemBackend` raises a path escape
    with its host root in the message.
  - The example's directory carries the benchmark's name, which PEtabAgentBench's `ANON_MODEL_ID`
    exists to keep from the agent.
  - File-tool errors are also scrubbed of the host root.
- **`/problem/` is read-only in three ways:**
  - for the shell, `--ro-bind` over `/work/problem`;
  - for the file tools, the backend refuses `write_file` and `edit_file` under `/problem/`. The
    message says the copy is the server's and changes go through `make_changes`;
  - on the host, the files are `0444`.
- **deepagents' own `FilesystemPermission` rules cannot do the file-tool part.** With a backend
  that executes, they raise `NotImplementedError` unless the rule's path is a `CompositeBackend`
  route. So the refusal is a method override on the existing `_BwrapSandbox`, which keeps one
  backend class for both arms.
- **Everything else is scratch,** shared by the panel.

**What writing means.** A scratch file is the agent's notebook: nothing reads it except the agent
and, after the run, the archive. In particular:

- the server is started on `work/problem.yaml`, never on the workspace, and nothing copies a
  workspace file into `work/`;
- the judge (`runner.judge._judge`) takes the MCP caller and `work/`, and scores the server's best
  version. It is never handed the workspace path;
- a model the agent wants scored must be submitted as text to `make_changes`, which fits, versions
  and records its justification like any other.

After the run, the scratch files are archived beside the result in `<archive>/workspace/`. The
size cap and suffix list `runner.barebones._archive_delivery` applies to barebones apply here too.
`problem/` is not archived, since the version store already holds every version.

## 3. Code execution

### Reuse the sandbox

`execute` is `sandbox.sandbox_backend_class()`, as on barebones:

- bwrap, with the Python environment and system trees read-only and the workspace read-write at
  `/work`;
- no network and no `$HOME`;
- an allowlisted environment and its own process group;
- every command recorded in the shell audit.

Two additions to `sandbox.py`, both usable by barebones:

- `bwrap_argv` takes read-only binds (`--ro-bind <ws>/problem /work/problem`, after the workspace
  bind);
- `_BwrapSandbox` takes read-only path prefixes for its file tools.

`sandbox_preflight` runs before the first LLM call, as on barebones. On a node where it fails, the
run writes a degraded result (`sandbox: ...`). There is no unsandboxed fallback.

One command may run for `fit_time_budget(n_starts)`, as on barebones. The driver's stall window
must cover a pending `execute` the way it covers a pending fit. Today `_pending_fit_starts` sizes
it for `make_changes` only.

### What the shell cannot reach

The panel arm adds one thing barebones never had to consider: a live server on the same node.

- **No route to the server.** `--unshare-all` gives the sandbox its own network namespace, so
  `127.0.0.1:<port>` inside is not the server. The shell cannot call `evaluate_validation_data`,
  `restore_version` or any other tool outside the agent's curated list. A test proves this
  (section 6).
- **No view of the server process.** The sandbox has its own PID namespace, so `/proc` shows no
  server process. The server's command line names the held-out table's path.

### One package set for both arms

Today three places define the set, and they can drift:

- the environment itself: `sandbox.default_env_prefix()`, the interpreter the harness runs from;
- the prose list in `prompts.BAREBONES_SYSTEM_PROMPT_TMPL`: "amici, pypesto (with the fides
  optimizer), petab, python-libsbml, antimony, numpy, scipy, pandas, sympy, h5py and
  scikit-learn";
- the dependencies in `pyproject.toml`. Their comment says pyPESTO is "promised to the barebones
  agent's sandbox", but h5py and scikit-learn are not among them.

Proposal:

- **One tuple, `SANDBOX_PACKAGES`**, holding each package's import name and the name the prompt
  shows. It lives in a module that both `prompts.py` and `sandbox.py` can import without the agent
  stack (`test_import_lightness`).
- **Both prompts render the list from it,** the barebones prompt and the workspace prompt. A
  package dropped from the tuple disappears from both.
- **`sandbox_preflight` imports each package inside the sandbox** and refuses the node if one
  fails. The promise is then checked where it is made.
- **Each package is a direct dependency in `pyproject.toml`,** with a why-comment, per the style
  guide's "A new import means a new pin". A test checks the tuple against `[project].dependencies`.

Both arms run from the same environment prefix, so with one tuple their package sets cannot drift.
Other packages are importable too (petab_mcp, gemot, langchain), because the sandbox binds the
whole prefix. See open question 6.

This refactor must not change the barebones prompt's rendered text, so
`prompt_group_versions()["barebones"]` stays as it is. A test asserts the hash.

### Node resources

On barebones, the server is not running while the agent works. Here the sandbox shares the node
with the server's fits, and both get `FIT_N_WORKERS` cores. A local multistart during a server fit
slows both down, and the fit budgets in `fit_budget.py` assume the cores belong to the fit. See
open question 7.

### No token limit

Nothing here touches a model call. No `max_tokens` and no cap of any kind is added to any call,
retry or hand-off. The sandbox's existing 100 kB bound on one command's output is a bound on tool
output, as on barebones.

## 4. Agents, tools and prompts

### Who holds what

| Agent | Today | On the workspace arm |
|---|---|---|
| lead | `ORCHESTRATOR_TOOLS` plus `finish`; `_builtin_tool_filter` strips deepagents' built-ins | the same, plus `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` and `execute`; the filter strips only `write_todos` |
| mathematician, systems_scientist, signal_engineer | their curated MCP tools, plus deepagents' built-ins over an empty in-state filesystem (`subagents.py`: they cannot be filtered) | the same built-ins, now over the workspace |
| biocurator | bio tools and `list_observables`, plus the same unfilterable built-ins | **no file tools and no `execute`**: its blind excludes the measurements, and the copy has them |
| `general-purpose` | disabled | disabled |

`ALLOWED_TOOLS` and the persona lists in `tools.py` curate MCP tools and do not change. The
built-ins come from the backend.

**The biocurator's exclusion has to be structural.**

- deepagents 0.7.1 gives every declarative subagent a `FilesystemMiddleware` over the parent's
  backend, and reads no `backend` key from the spec.
- So the biocurator becomes a `CompiledSubAgent`: a graph gemot builds itself with the
  biocurator's tools, its turn limit and its runaway guard, and no filesystem middleware.
  deepagents uses such a graph as it is.
- A test reads the tool names off the built graph.

**Guards.** The dispatch guard's `tools_executed` counts built-in calls as well. `no_model_built`
still keys on commits. The circuit breaker keeps wrapping MCP tools only.

### Prompts

New text goes into a new `PROMPT_GROUPS` group, `workspace`. A change to it is then attributable,
and the other groups' versions do not move.

- **`WORKSPACE_TMPL`** is one constant. It is composed into the lead's system prompt
  (`DEFAULT_SYSTEM_PROMPT`, or `INDRA_SYSTEM_PROMPT` on that arm) and into each analyst's persona,
  with the agent's name filled in. It states:
  - the layout: `/problem/` is the server's problem at HEAD, read-only, and `VERSION` says which
    version it is. The rest is scratch shared with the panel, and each agent writes under a
    directory named for itself;
  - the shell: `execute`, the cwd `/work`, no network, the packages rendered from
    `SANDBOX_PACKAGES`, the command timeout, the cores, and the background-job advice barebones
    gives;
  - the v2 file description and the `parameterScale` sentence. Both become constants shared with
    `BAREBONES_PETAB_FORMATS["v2"]`, lifted out of it without changing its text;
  - **the rule, as one constant, `_WORKSPACE_IS_NOT_SCORED`**: nothing written here reaches the
    server or the judge, the model changes only through `make_changes`, and the only score is the
    server's.
- **The lead's "You directly read only…" sentence becomes false.** It says the lead reads "only the
  model, its SCORES … for the raw DATA … you DELEGATE to the analysts", which no longer holds once
  the lead can `read_file`. On the workspace arm the lead prompt composes a variant instead:
  - the lead may draft and test model text and run short checks in the workspace;
  - bulk reading stays with the analysts, for the same token reason as today;
  - the off-arm text stays byte-identical.
- **`_REPORT_CONTRACT` stays true.** Its "You cannot edit the model" still holds. The workspace
  block adds that an analyst can write and run scripts.
- **Unchanged:** the task, the biocurator, the finish gate, the nudges and the tool descriptions.

## 5. How it lands

### An ablation, not a driver

It is the `workspace` ablation (`PETAB_WORKSPACE`, default off, `changes_prompt=True`).

- **It meets `ablations.py`'s membership test**: it changes "what tools it holds".
- **It follows the one-mechanism rule**: every arm is an environment variable. A new `--driver`
  would be a second mechanism and could not combine with the existing ablations.

It combines with `biocurator`, `reference_handoff`, `bio_literature` and `indra_statements`. It is
refused at startup in three cases:

- **with `biology_blind`**, in the first version, as barebones refuses it. Under the export the
  copy would already be anonymous, since the server serves anonymous names. The neutral path and
  the error scrubbing are in this plan. Open question 5 asks what else would be needed;
- **on `--driver barebones`**, which always has a workspace;
- **on `--driver scripted`**, which has no agent.

So a recorded `workspace=on` always means a panel with a workspace.

### Comparability with earlier sweeps

- **Off is the old configuration.** It has the same tools, the same prompts and the same server
  behaviour. `ablation_arm` stays `baseline`, because `arm()` names only deviations.
- **`prompt_version` moves for every run.** It hashes every registered prompt, used or not. Compare
  across the change by `prompt_group_versions()`: `lead`, `analysts`, `barebones` and the other
  groups must not move. A test asserts this for `barebones`.
- **`state_key()` moves,** as for any appended switch. This is the trade its docstring documents.
- **The server pin moves** for the export resource, and brings whatever else is on PEtab-MCP main
  by then. So earlier sweeps are not the control. The A/B is three arms at one gemot commit and one
  server pin: panel off, panel on, barebones. That is the rule the retired-ablations entry in
  `docs/RATIONALE.md` states ("a fresh A/B with both arms on one code version").
- **`Scenario.version` does not move,** because no example changes.

## 6. Checks

The checks live in a new `tests/test_workspace.py`, plus additions to `test_sandbox.py`,
`test_ablations.py` and `test_prompt_tracking.py`. Each names the leak or regression it locks out.

**Isolation and leaks:**

1. **The copy is the served problem and nothing else.** The fixture example has `reference.json`,
   `validation_v1_baseline.json`, `validation/`, `validation_source/`, `indra/`, a `run.py`
   carrying a `TRUE_ANT` marker, `.amici_build/` and `work/run_live.json`. The workspace must hold
   only the copy's files and the sandbox's own directories, and no file in it may contain the
   reference `nll` or the marker.
2. **No held-out row in the export.** Checked on a server started with `validation_measurements`
   (a PEtab-MCP test).
3. **The sandbox sees nothing above the workspace**: not the example directory, `work/`, the build
   root, the blinding directory or `$HOME`.
4. **The sandbox cannot reach the server.** A connect to the server's port from inside fails, and
   `/proc` lists no server.
5. **`/problem/` refuses writes** from the shell (EROFS) and from `write_file` and `edit_file` (the
   refusal message). A symlink or `..` out of the workspace is refused too: deepagents' own check,
   asserted here.
6. **No host path in any tool output.** The example's name does not appear in a file-tool error.
7. **No problem tree in the environment prefix.** The preflight refuses a non-editable
   PEtabAgentBench install, because its wheel ships every problem's `run.py` and `reference.json`
   under `petab_agent_bench/problems/` (`hatch_build.py`). gemot_iclr_2026's README installs it
   editable, so this guards a future install, not today's.

**Behaviour:**

8. **The copy follows HEAD.** After a scripted commit on the toy problem, `/problem/model.ant`
   equals `get_model()`, `parameters.tsv` equals the PEtab columns of `get_parameters()`, and
   `VERSION` is HEAD. A refused commit leaves all three untouched.
9. **Scoring stays the server's.** A run whose scratch holds a better model than the server's best
   is judged on the server's best.
10. **Who holds what.** The lead and the analysts hold the file tools and `execute`, the biocurator
    holds neither, and `general-purpose` is absent. Read off the built graph, in a subprocess,
    because the arm is resolved at import time (style guide).
11. **One package set.** The preflight imports every `SANDBOX_PACKAGES` entry, each is a
    `pyproject.toml` dependency, and both prompts format from the tuple.
12. **Prompt tracking.** The `workspace` group is registered. The off-arm lead and analyst prompts
    and the `barebones` group hash are unchanged.
13. **The ablation** is registered and recorded, and refused with `biology_blind` and on the
    barebones and scripted drivers.
14. **`result.json`** carries `sandbox`, `workspace_audit` (barebones' keys, so one query reads
    both arms) and the copy's refresh and failure counts. The archive holds the scratch files.
15. **The stall window** covers a pending `execute`.

**CI.** The bwrap tests in `test_sandbox.py` are guarded by `requires_bwrap`, and CI installs no
bubblewrap, so they skip there today. Every check above would then go unrun in CI.

- The workflow installs `bubblewrap` and allows unprivileged user namespaces
  (`kernel.apparmor_restrict_unprivileged_userns=0` on Ubuntu 24.04 runners), so these tests run.
- This is a separate PR, landing first.

**Cluster smoke run.** Before any sweep, run the toy problem and then one benchmark. Each runs a
probe inside the sandbox: `find / -name reference.json`, and a connect to the server's port.

## 7. Rollout

| Step | Repo | What | Pins |
|---|---|---|---|
| 1 | gemot | CI runs the bwrap tests (bubblewrap, user namespaces) | none |
| 2 | gemot | `SANDBOX_PACKAGES`, the preflight's import check, h5py and scikit-learn as dependencies, the shared v2 prompt constants; no prompt text changes | none |
| 3 | PEtab-MCP | the `petab://problem/files` resource and its tests | none |
| 4 | gemot | the `workspace` ablation: `agent/workspace.py` (layout, export, swap, archive), `sandbox.py` (read-only binds, read-only prefixes, error scrubbing), the driver wiring, the compiled biocurator, the `workspace` prompt group, the tests | `petab-mcp` to the first main commit with step 3 |
| 5 | none | cluster smoke run with the in-sandbox probe | none |
| 6 | gemot_iclr_2026 | bump the gemot pin, add the arm to `slurm/make_sweep.py`, run the three-arm A/B | `gemot` to step 4 |

**PEtabAgentBench needs no change.** Its problems already keep the held-out sets and answer keys
out of what the server loads. Its `gemot` extra pins gemot only for materialization and refits, and
can follow whenever.

Step 4 adds at most one `docs/RATIONALE.md` entry: why the copy is the server's export rather than
a filtered copy of `problem/`. The barebones leak below is the failure that motivates it.

## 8. Open questions for reviewers

1. **Does the lead get the workspace?** Proposed: yes, for parity with barebones, whose single
   agent both edits and runs code. Against: the lead's tool list is curated to keep bulk reads out
   of its re-sent history, and `read_file` on `measurements.tsv` is such a read. The alternative is
   to give it to the analysts only.
2. **Server export or gemot-only?**
   - The gemot-only version copies the design tables once, through the server's own loader
     (`WorkingProblem.from_yaml`, as `biology_blinding.prepare` does).
   - After each commit, it writes `model.xml` and `model.ant` from `get_model`, and `parameters.tsv`
     from the records `get_parameters` returns.
   - That needs no PEtab-MCP change. But gemot becomes a second writer of the server's parameter
     table: the reconciled rows, the `estimate` spellings, the scale column. The export avoids
     that.
3. **HEAD only, or every version** (`/versions/vN/`) and the HEAD fit's simulation table? Proposed:
   HEAD only, since `get_version` and `query_data` already serve the rest.
4. **Is an incomplete version history acceptable?** With a local pyPESTO, the versions no longer
   record the whole search. Proposed: accept it, record it (audit and archive), and say in the
   prompt that local numbers are not scores. The alternatives are to count local fits from the
   audit, or to discourage them.
5. **Support `biology_blind` later?** The export would already be anonymous, and the neutral path
   and error scrubbing are in this plan. What remains is a review of what the sandbox's packages
   could reveal; we know of nothing.
6. **The package set: the harness environment, or exactly the tuple?**
   - The sandbox binds the whole prefix, so petab_mcp, gemot and langchain are importable. That is
     true on barebones too, whose prompt says "no server, no helper library".
   - A dedicated environment holding only `SANDBOX_PACKAGES` is the strict reading of "the same
     set". It costs a second environment to build and pin.
7. **Cores:** should the sandbox and the server's fits share the node, or should the cores be split
   (and the fit budgets with them)?
8. **Analyst turn limit:** is `ANALYST_MAX_TURNS = 20` still enough once an analyst can write and
   run a script?
9. **Shared scratch:** per-agent directories by convention, as proposed, or a separate workspace
   per delegation?

## Found while reading

These are separate fixes, not part of this plan.

- **Barebones stages an answer key for five problems.**
  - `runner.barebones.stage_workspace` withholds files by name:
    `_WITHHELD = ("reference.json", "model.ant")`.
  - `problem/validation_v1_baseline.json` holds the reference's held-out score. It is tracked in
    `problem/` for perelson, crauste, fujita, lucarelli and raimundez (PEtabAgentBench
    `.gitignore`), so it reaches the barebones agent's `work/agent/`.
  - The fix is an allowlist built from `problem.yaml` (`_design_files` already reads it), plus
    `inputs.tsv`.
  - Whether earlier barebones runs on those five problems read the file can be checked in their
    `workspace_audit.commands`.
- **CI skips the sandbox tests**, because the runner has no bubblewrap. This is step 1 above.
- **The barebones prompt promises packages the dependencies do not name** (h5py, scikit-learn).
  This is step 2 above.
- **Barebones' workspace path names the example** (`examples/<name>/work/agent`). deepagents'
  path-escape error carries that path. Not checked: whether the tool layer passes that message on
  to the agent.

## Not in this plan

- `write_todos` and deepagents' `general-purpose` subagent. Barebones has both and the panel does
  not, but neither is the filesystem or the package set.
- Any change to the editing loop itself, the finish gate or the judge.
