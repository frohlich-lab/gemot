# petab-mcp

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns model
building into a **disciplined, auditable** workflow — the loop behind model-building
assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

- The experimental design (**measurements, conditions, observables**) is read-only.
- The **model is authored and edited only in Antimony**, through constrained, justified
  operations: **author/replace the whole model** (`set_model`), **add a reaction, change a
  rate law, remove a reaction**, and adjust a parameter's **scale/bounds** — never free-form.
- Every operation **must cite a justification**; references are **optional** (validated
  against **PubMed** when provided).
- Operations **auto-maintain SBML↔PEtab consistency**: new symbols become model
  parameters or species, orphans are pruned, and **every species uses a parametric
  initial condition** (`S = S_init`), so the parameter table always matches the model.
- Every commit **automatically fits + simulates**: it estimates the free parameters,
  simulates against the fixed data, and stores a **simulation dataframe** on the new
  version, so statistical analysis reads a persisted result instead of re-simulating.
- Every change is **versioned and logged** (provenance): each version records the fitted
  parameters, fit metrics, and simulation dataframe; alterations and rollbacks are
  persisted to disk with full metadata.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI); optimization uses
[pyPESTO](https://github.com/ICB-DCM/pyPESTO) with the [Fides](https://github.com/fides-dev/fides)
trust-region optimizer. All are **required** — fitting runs on every commit.

## Install

```bash
pip install -e .          # installs AMICI + pyPESTO (a C++ toolchain is required)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and
the pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

## Run

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# options: --provenance-dir DIR   --reference-validation {required,best_effort,off}
```

The canonical model is Antimony: if a `model.ant` sits next to the YAML it is the
source of truth. A problem may also ship **without** a model (an empty SBML skeleton and
no `model.ant`); the agent then zero-shots the whole model once from Antimony with
`set_model` (see below). Provenance is written to `<problem_dir>/.petab_mcp/` by default
and is **durable** — restarting resumes the latest version. A ready toy problem (reversible
conversion reaction A⇌B) lives at `tests/data/toy_v1/`.

## Examples

[`examples/`](examples/) holds end-to-end **zero-shot model-building demos**: an
OpenRouter-hosted LLM, driving a panel of three specialist analyst subagents (via
[`deepagents`](https://github.com/langchain-ai/deepagents)) through these tools, is handed a
PEtab problem that **ships without a model** (an empty SBML) and has to **build** the
mechanism from the data — one `set_model` call, then refinement. Each has an obvious
**trap** (a linear/bilinear/first-order rate law that can't reproduce the dynamics for any
parameters), so fitting the numbers isn't enough; an independent judge checks the built
**mechanism** by BIC against the true model, not just the fit. Every example also has a
no-LLM `--driver scripted` self-test.

| Example | System | The trap → the true mechanism |
|---|---|---|
| [`enzyme_kinetics/`](examples/enzyme_kinetics/) | Two-reaction enzymatic network, two doses | Michaelis–Menten → **substrate inhibition** (`…/(Km+S+S²/Ki)`) |
| [`predator_prey/`](examples/predator_prey/) | Rosenzweig–MacArthur limit cycle | Lotka–Volterra bilinear → **Holling type-II** saturating response (2 coupled fluxes) |
| [`repressilator/`](examples/repressilator/) | Three-gene ring oscillator | Non-cooperative Hill `n=1` → **cooperative** `n≥2` (3 symmetric reactions) |

See [`examples/README.md`](examples/README.md) for the shared harness (the analyst panel,
Arize Phoenix tracing, judging, and how to run).

## Tools

**Read-only (the fixed design)** — `describe_problem`, `list_observables`,
`list_conditions`, `get_measurements(condition_id?, observable_id?)`.

**Model operations** (each requires a `justification`; `references` are optional) —
- `set_model(antimony, justification, references?)` — set the whole model from an
  Antimony string. Builds the first model when none is initialized yet
  (`describe_problem` → `model_initialized: false`) and re-authors/replaces an existing model.
  New species/parameters + parametric ICs are reconciled automatically.
- `change_parameter_scale(parameter_id, scale, justification, references?)` — references optional.
- `change_parameter_bounds(parameter_id, lower_bound, upper_bound, justification, references?)`.
- read-only: `get_model(format)`, `get_model_structure`, `validate_problem`.

**Batching** — `begin_batch(label)` → run several operations (applied + logged, not yet
versioned) → `commit_batch(label)` snapshots them as **one** version (fitted once);
`abort_batch` discards staged edits. `rollback_to`/`reset` are blocked while a batch is open.

**Parameters** — `get_parameters` (read-only; values are set by the fit), `sync_parameters`.
Parameter *values* are set only by the automatic fit; adjust the search space with the
versioned `change_parameter_bounds` / `change_parameter_scale` operations. Every commit returns
an `initial_conditions` block reporting, per species, whether its declared t=0 is used (fixed),
fitted (estimated), or set per-condition.

**Fit + simulation (automatic on commit)** — every commit fits the free parameters
(pyPESTO multistart) and simulates, then snapshots a single version carrying the fitted
parameters, the fit metrics (`neg_log_likelihood`, `chi2`, `bic`), improvement metrics
(`objective_before`, `self_improvement`, `improvement_vs_best_prior`, `is_new_best` — so
versions can be ranked; `is_new_best`/`improvement_vs_best_prior` track **BIC** against the
best model so far, and `is_new_best` is gated on a minimum effect size, ΔBIC ≥ 2, so
noise-floor wander — or an nll gain that raises BIC — isn't counted as progress), and the
simulation dataframe. Each commit result also carries a `progress` block (non-improving
streak, best-BIC `best_version`, best-BIC gain over the last N ops) that backs an early-stop
once the score stops moving — see `get_progress`. Pass `n_starts` on a commit to fit harder
(`0` = simulate without fitting); batching fits once on `commit_batch`.

**Fit inspection** — `get_fit_summary()` returns just the metrics (no records; a cheap
score check). Three compact, size-bounded helpers give a quick read without pulling the raw
data: `get_fit_diagnosis()` (per-observable χ²/bias + the worst-fit points),
`get_identifiability()` (which estimated parameters are unconstrained or trade off — the basis
for simplifying an over-parameterised model), and `get_run_timing()` (cumulative server-side
compile/fit wall times). For anything else the fit is **not** pre-digested into a fixed menu —
it is **SQL over the data** via DuckDB, so you compute the breakdown you want:
- `list_data_tables()` lists the queryable tables and their columns; `query_data(sql,
  max_rows?)` runs a DuckDB query against a **disposable copy** of the tables — any SQL is
  allowed and cannot affect the session (host file/network access is disabled) — and returns
  `{columns, rows, row_count, truncated}`.
- Tables: the fixed design (`measurements`, `conditions`, `observables`), the `parameters`
  table, and the `simulations` dataframe from the last commit — one row per measurement
  point with `measurement`, `simulation`, the raw residual (`residual` = `measurement -
  simulation`, in observable units), the **noise-weighted** residual (`residual_weighted` =
  `(m - s) / σ` on the observable scale — the χ² term, reproducing PEtab's `calculate_chi2`),
  and the evaluated noise σ (`noise`).
- The last fit's **pyPESTO multistart** is queryable too: `optimization_starts` (one row per
  start, `start`=0 is best — `fval`, `time`, `n_fval`/`n_grad`/`n_hess`, `exitflag`, `message`)
  and `optimization_trace` (the objective `fval` per `start`/`iteration`), so convergence and
  multistart spread are a SQL query away (`SELECT count(*), min(fval), max(fval) FROM
  optimization_starts`). These are **persisted per version** (as `optimization.json` plus a
  pyPESTO `pypesto_result.hdf5` result file, cached per model by SBML hash and copied into the
  version dir), so rolling back to a version restores its traces — not just its model. Empty
  on a simulate-only pass.
- Residual statistics, per-observable/-condition/-timepoint breakdowns, worst-fit points,
  and bias or time-trend checks are all just SQL — filter to `residual_weighted IS NOT NULL`
  for residual stats (a NULL residual, from an unsimulated point or un-evaluable noise model,
  would inflate a `count(*)`), e.g. per-observable χ²
  (`SELECT observable_id, count(*) AS n, sum(residual_weighted*residual_weighted) AS chi2
  FROM simulations WHERE residual_weighted IS NOT NULL GROUP BY observable_id ORDER BY chi2
  DESC`) or the worst points (`SELECT * FROM simulations WHERE residual_weighted IS NOT NULL
  ORDER BY abs(residual_weighted) DESC LIMIT 10`). The `simulations` table is generated on
  demand at cold start (no commit yet).
- The **pyPESTO/Fides optimizer log** is exposed separately by `get_fitting_log(tail_lines?)`
  (and the `petab://fit/log` resource): multistart convergence, parameters pinned at bounds,
  and ill-conditioning — the *numerical* view of the fit, complementing the SQL residual view.

**Provenance, versions & errors** — `get_provenance`, `get_errors`, `list_versions`,
`get_version`, `diff_versions`, `rollback_to(version)`, `restore_version(version)`,
`current_version`, `reset`.

**Skills** — `list_skills`, `get_skill(name, file?)`. The server ships tool-usage
documentation as [agent skills](https://modelcontextprotocol.io) bundled under
`src/petab_mcp/_skills/` and exposes them over MCP so a client can learn the workflow
without leaving the session (see below).

Resources: `petab://model/antimony`, `petab://model/sbml`, `petab://observables`,
`petab://conditions`, `petab://measurements`, `petab://parameters`,
`petab://provenance`, `petab://versions`, `petab://errors`, `petab://simulations`,
`petab://data/schema`, `petab://fit/log`, `petab://skills`, `petab://skills/{name}`.

## Skills

`src/petab_mcp/_skills/` holds self-describing **skills** — markdown docs that explain
how to use this server's tools — in the standard agent-skill layout (a directory per
skill with a `SKILL.md` frontmatter header plus supporting files). Bundling them inside
the package means they resolve identically from a source checkout and an installed
wheel. The `petab-model-building` skill documents the disciplined edit→score→audit
workflow.

Skills are exposed to MCP clients as:

- tools — `list_skills()` and `get_skill(name, file?)` (the reliable path, since
  every client consumes tools);
- resources — `petab://skills` (a JSON index) and `petab://skills/{name}` (a skill's
  `SKILL.md`).

Override the directory with `--skills-dir DIR` or `PETAB_MCP_SKILLS_DIR` to point at
your own skills; drop another skill directory into `_skills/` to extend the bundled docs.

## Design notes

- **Constrained, justified editing.** There is no free-form model-replacement or
  SBML-input tool. Each operation runs a commit pipeline: validate references → apply to
  a model copy → reconcile SBML↔PEtab → lint → commit or revert → snapshot + log. An
  edit that breaks a fixed observable/condition is rejected.
- **Parametric initial conditions.** `reconcile` guarantees every species `S` has an
  `initialAssignment(S := S_init)` with `S_init` in the parameter table, so initial conditions are
  first-class parameters. They are **estimated by default** — a declared t=0 is only a guess and
  must never be silently frozen (a fixed guess the optimizer can't correct is the trap behind #90).
  Two leak-free *value conventions* stay **fixed** instead: an IC of exactly `0` (an empty pool,
  fixed at 0) and an IC of exactly `1` (a unit normalization — e.g. a conserved-total set to 1 —
  fixed by value, so it names no species). To keep any other IC fixed, ship it as a fixed parameter
  row (reconcile never re-touches existing rows). Each commit's `initial_conditions` block reports
  each species' actual mode.
- **PubMed validation.** References are **optional** on every operation (only a non-empty
  `justification` is required); when given, they are resolved via NCBI E-utilities (no extra
  dependency). The mode governs how strictly a *supplied* reference is validated: `required`
  rejects one that cannot be resolved; `best_effort` (default) records a `verified` flag and
  never blocks on the network; `off` skips validation.
- **Provenance, versioning & error logging.** `.petab_mcp/provenance.jsonl` is an
  append-only event log; `.petab_mcp/versions/vN/` snapshots the Antimony model +
  parameter table (as a PEtab `parameters.tsv`, not JSON), and for a fitted version its
  `simulations.json`, `optimization.json` (pyPESTO traces), and `pypesto_result.hdf5` (the
  pyPESTO result file, staged in `.petab_mcp/cache/` and copied in); `.petab_mcp/cache/` also
  holds the fit cache (`fit_<key>.json`) keyed by estimation-problem identity, letting an
  identical future commit reuse a fit; `.petab_mcp/errors.jsonl` records
  rejected operations / failed tool calls (surfaced via `get_errors`); `.petab_mcp/petab_mcp.log`
  is a human-readable log (including a concise per-commit fit summary), and
  `.petab_mcp/pypesto.log` captures pyPESTO's own multistart optimizer report for each fit.
  `rollback_to` restores a prior version as a new, auditable version — including its fit
  artifacts (simulation + optimization traces + result file), copied forward. Batching chains
  several operations into a single version.
- **Fit on commit + comparison.** Every commit fits and records its objective and how it
  compares to the best prior version, so competing model versions can be ranked by fit
  without a separate optimize step.
- **Cheap parameter edits, cached compilation.** AMICI models are cached by SBML hash;
  a structural operation recompiles, parameter edits do not.
- **Fit reuse (cache).** Every fit is cached under its *estimation-problem* identity — the
  model plus the parameter table (bounds, scale, which parameters are estimated, fixed
  values) plus the multistart count and seed, but **not** an estimated parameter's starting
  value (which the multistart optimizer does not use). Committing an estimation problem that
  was already fit reuses the stored fit (fitted parameters + traces + pyPESTO result) and
  skips the expensive optimization; the model is still simulated, so the simulation dataframe
  and metrics are always current. The result is otherwise identical, so this is transparent.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest
pytest                    # AMICI/pyPESTO fit tests skip if absent; PubMed path is stubbed
```

## License

MIT
