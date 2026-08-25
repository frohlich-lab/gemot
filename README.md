# petab-mcp

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns model
building into a **disciplined, auditable** workflow — the loop behind model-building
assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

- The experimental design (**measurements, conditions, observables**) is read-only.
- The **model is authored and edited only in Antimony**, through one constrained, justified
  operation: **`make_changes` is the single editor** — it authors the whole model (every
  structural change is a whole-model re-author) and/or sets a parameter's **scale/bounds**,
  never free-form.
- Every operation **must cite a justification**; references are **optional** (validated
  against **PubMed** when provided).
- Operations **auto-maintain SBML↔PEtab consistency**: new symbols become model
  parameters or species, orphans are pruned, and **every species uses a parametric
  initial condition** (`S = S_init`), so the parameter table always matches the model.
- **Every change is scored.** `make_changes` fits and snapshots a version on every call — there
  is no fit-free edit and no standalone fit, so a change and its score cannot come apart. One
  coherent change is one call.
- Every change is **versioned and logged** (provenance): each version records the fitted
  parameters, fit metrics, and simulation dataframe; alterations and rollbacks are
  persisted to disk with full metadata.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI); optimization uses
[pyPESTO](https://github.com/ICB-DCM/pyPESTO) with the [Fides](https://github.com/fides-dev/fides)
trust-region optimizer. All are **required** — fitting is not optional machinery.

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
no `model.ant`); it is seeded with a **content-free initial model** — one species per identifier
the fixed observable/condition tables already name, no reactions — so v0 is a valid, simulatable,
restorable version rather than a broken empty skeleton, and `describe_problem` reports
`model_authored: false` until the agent zero-shots the mechanism with `make_changes` (see below). Provenance is written to `<problem_dir>/.petab_mcp/` by default
and is **durable** — restarting resumes the latest version. A ready toy problem (reversible
conversion reaction A⇌B) lives at `tests/data/toy_v1/`.

## Examples

[`examples/`](examples/) holds end-to-end **zero-shot model-building demos**: an
OpenRouter-hosted LLM, driving a panel of three specialist analyst subagents (via
[`deepagents`](https://github.com/langchain-ai/deepagents)) through these tools, is handed a
PEtab problem that **ships without a model** (an empty SBML) and has to **build** the
mechanism from the data — one `make_changes(antimony=…)` call, then refinement. Each has an obvious
**trap** (a linear/bilinear/first-order rate law that can't reproduce the dynamics for any
parameters), so fitting the numbers isn't enough; an independent judge checks the built
**mechanism** by BIC against the true model, not just the fit. Every example also has a
no-LLM `--driver scripted` self-test.

There are **16** examples. Three are synthetic (they simulate their own data from a withheld
truth model, so the trap is planted deliberately):

| Example | System | The trap → the true mechanism |
|---|---|---|
| [`enzyme_kinetics/`](examples/enzyme_kinetics/) | Two-reaction enzymatic network, two doses | Michaelis–Menten → **substrate inhibition** (`…/(Km+S+S²/Ki)`) |
| [`predator_prey/`](examples/predator_prey/) | Rosenzweig–MacArthur limit cycle | Lotka–Volterra bilinear → **Holling type-II** saturating response (2 coupled fluxes) |
| [`repressilator/`](examples/repressilator/) | Three-gene ring oscillator | Non-cooperative Hill `n=1` → **cooperative** `n≥2` (3 symmetric reactions) |

The other thirteen are **real published models** with real experimental data, materialized from
the pinned PEtab Benchmark-Models collection by `python examples/setup.py` (armistead, blasi,
boehm, brannmark, bruno, crauste, fiedler, fujita, oliveira, perelson, rahman, schwen, sneyd).
Each ships model-less, so the agent must build the mechanism from the data; the BIC bar is the
published model's own fit. See [`examples/BENCHMARK_NOTES.md`](examples/BENCHMARK_NOTES.md).
The roster lives in one place, `petab_harness.paths` — add a new example there.

See [`examples/README.md`](examples/README.md) for the shared harness (the analyst panel,
Arize Phoenix tracing, judging, and how to run).

## Tools

**Read-only (the fixed design)** — `describe_problem`, `list_observables`,
`list_conditions`, `get_measurements(condition_id?, observable_id?)`.

**The editor** — `make_changes(justification, antimony?, parameters?, references?, n_starts?)` is
the **single** tool that changes anything, and it always fits:
- `antimony` sets the whole model from an Antimony string. It builds the first model when none is
  initialized yet (`describe_problem` → `model_authored: false`) and re-authors/replaces an
  existing one, so every structural change (adding or removing a reaction, changing a rate law) is
  one coherent re-author. New species/parameters + parametric ICs are reconciled automatically.
- `parameters` batch-edits the bounds and/or scale of one or more parameters: a non-empty list of
  `{parameter_id, lower_bound?, upper_bound?, scale?}` entries (a bound edit gives **both**
  bounds).
- Give either, or both — but not neither. Both land in the **same transaction**, applied after the
  reconcile, so an entry may target a parameter the re-author introduces, and one rejected entry
  reverts the whole call.
- read-only companions: `get_model(format)`, `get_model_structure`.

Every call then fits and snapshots exactly one version, so a change and its score cannot come
apart. The result **is** the fit — the same field names `get_fit_summary` uses — plus `applied`
(is the change in the model?) and `committed` (did it make a version?). If the edit applies but
the fit fails, the edit is **reverted** (`applied: false` + `error`) — nothing is left half-done.

**Parameters** — `get_parameters` (read-only; values are set by the fit).
Parameter *values* are set only by the fit; adjust the search space with `make_changes`'
`parameters` list. `get_model_structure` carries the per-species `initial_conditions` account —
whether each declared t=0 is used (fixed), fitted (estimated), or set per-condition.

**The fit** — every `make_changes` call fits
the free parameters (pyPESTO multistart), simulates, and snapshots a single version carrying the
fitted parameters, the fit metrics (`neg_log_likelihood`, `chi2`, `bic`), improvement metrics
(`objective_before`, `self_improvement`, `improvement_vs_best_prior`, `is_new_best` — so
versions can be ranked; `is_new_best`/`improvement_vs_best_prior` track **BIC** against the
best model so far, and `is_new_best` is gated on a minimum effect size, ΔBIC ≥ 2, so
noise-floor wander — or an nll gain that raises BIC — isn't counted as progress), and the
simulation dataframe. The result also carries a `progress` block (non-improving
streak, best-BIC `best_version`, best-BIC gain over the last N ops) that backs an early-stop
once the score stops moving — see `get_progress`. `n_starts` **replaces** the server's configured
multistart count (`create_server(auto_fit_starts=…)`, which the examples harness sets from its own
`--n-starts`; the standalone CLI uses the built-in default), so pass it only to go *above* that
default — a smaller value fits less hard, and the minimum is `1`. Neither the number nor
its effect has to be guessed: `describe_problem` reports the default as `fit_n_starts_default` and
every fit result reports the `n_starts` it actually ran with. The fit de-duplicates: a model
byte-identical to the current
version returns the stored metrics without refitting (`committed: false, no_change: true`), a
structure already elsewhere in the history is fit but **warns** to use `restore_version(n)`
instead, and the same structure fit repeatedly is **refused**. A model that can't compile or has
no free parameters returns `{committed: false, error}` and leaves the working model in place to
repair.

**Fit inspection** — `get_fit_summary()` returns just the metrics (no records; a cheap
score check). Two compact, size-bounded helpers give a quick read without pulling the raw
data: `get_fit_diagnosis(sections?)` — `residuals` (per-observable χ²/bias + the worst-fit
points; the default), `parameters` (which estimated parameters are
unconstrained or trade off, and what each one-shared-value merge would cost — the basis for
simplifying an over-parameterised model — a local expansion around the LAST FIT's optimum, read off
that fit's own frozen record and, in a process that did not run it, off the verdict the fit stored,
so **no diagnosis can ever trigger a model build**), or `all` — the
cheapest few merges worth taking ride on
every fit result and on `get_fit_summary` as `merge_hints` — and `get_run_timing()` (cumulative server-side
compile/fit wall times). For anything else the fit is **not** pre-digested into a fixed menu —
it is **SQL over the data** via DuckDB, so you compute the breakdown you want:
- `list_data_tables()` lists the queryable tables and their columns; `query_data(sql,
  max_rows?)` runs a DuckDB query against a **disposable copy** of the tables — any SQL is
  allowed and cannot affect the session (host file/network access is disabled) — and returns
  `{columns, rows, row_count, truncated}`.
- Tables: the fixed design (`measurements`, `conditions`, `observables`), the `parameters`
  table, and the `simulations` dataframe from the last fit — one row per measurement
  point with `measurement`, `simulation`, the raw residual (`residual` = `measurement -
  simulation`, in observable units), the **noise-weighted** residual (`residual_weighted` =
  `(m - s) / σ` on the observable scale — the χ² term, reproducing PEtab's `calculate_chi2`),
  and the evaluated noise σ (`noise`).
- The last fit's **pyPESTO multistart** is queryable too: `optimization_starts` (one row per
  start, `start`=0 is best — `fval`, `time`, `n_fval`/`n_grad`/`n_hess`; the per-start
  `exitflag`/`message` are no longer columns — convergence is reported once per fit as an
  `exitflag_counts` histogram)
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
  demand at cold start (no fit yet).
- The **pyPESTO/Fides optimizer log** is exposed separately by `get_fitting_log(tail_lines?)`
  (and the `petab://fit/log` resource): multistart convergence, parameters pinned at bounds,
  and ill-conditioning — the *numerical* view of the fit, complementing the SQL residual view.

**Provenance, versions & errors** — `get_provenance`, `get_errors`, `list_versions`,
`get_version`, `diff_versions`, `restore_version(version)`, `get_progress`. (The HEAD
version comes from `describe_problem`.)

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
wheel. The `petab-model-building` skill documents the disciplined edit→fit→audit
workflow (and `antimony-authoring` covers writing the model text `make_changes` takes).

Skills are exposed to MCP clients as:

- tools — `list_skills()` and `get_skill(name, file?)` (the reliable path, since
  every client consumes tools);
- resources — `petab://skills` (a JSON index) and `petab://skills/{name}` (a skill's
  `SKILL.md`).

Override the directory with `--skills-dir DIR` or `PETAB_MCP_SKILLS_DIR` to point at
your own skills; drop another skill directory into `_skills/` to extend the bundled docs.

## Design notes

- **Constrained, justified editing.** There is no free-form SBML-input tool. Each edit runs
  one pipeline: validate references → apply to a model copy → reconcile SBML↔PEtab → lint →
  apply or revert → log. An edit that breaks a fixed observable/condition is rejected, and
  because the revert covers the model *and* the parameter table, a call that carries both an
  `antimony` re-author and a `parameters` list is atomic across both.
- **Parametric initial conditions.** `reconcile` guarantees every species `S` has an
  `initialAssignment(S := S_init)` with `S_init` in the parameter table, so initial conditions are
  first-class parameters. They are **estimated by default** — a declared t=0 is only a guess and
  must never be silently frozen (a fixed guess the optimizer can't correct is the trap behind #90).
  Two leak-free *value conventions* stay **fixed** instead: an IC of exactly `0` (an empty pool,
  fixed at 0) and an IC of exactly `1` (a unit normalization — e.g. a conserved-total set to 1 —
  fixed by value, so it names no species). To keep any other IC fixed, ship it as a fixed parameter
  row (reconcile never re-touches existing rows). `get_model_structure`'s `initial_conditions`
  block reports each species' actual mode.
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
  identical future fit be reused; `.petab_mcp/errors.jsonl` records
  rejected operations / failed tool calls (surfaced via `get_errors`); `.petab_mcp/petab_mcp.log`
  is a human-readable log (including a concise per-fit summary), and
  `.petab_mcp/pypesto.log` captures pyPESTO's own multistart optimizer report for each fit.
  `restore_version` restores a prior version as a new, auditable version — including its fit
  artifacts (simulation + optimization traces + result file), copied forward. A version is
  exactly a fit snapshot: a call appends an `edit` event and the fit it runs creates the version
  that scores it.
- **One version per coherent change + comparison.** Each fit records its objective and how it
  compares to the best prior version, so competing model versions can be ranked by fit
  without a separate optimize step — and because a whole change (re-author + bounds) is one
  call, it is scored once instead of appending a version per micro-edit.
- **Cheap parameter edits, cached compilation.** AMICI models are cached by SBML hash;
  a structural change recompiles, parameter edits do not.
- **Fit reuse (cache).** Every fit is cached under its *estimation-problem* identity — the
  model plus the parameter table (bounds, scale, which parameters are estimated, fixed
  values) plus the multistart count and seed, but **not** an estimated parameter's starting
  value (which the multistart optimizer does not use). Fitting an estimation problem that
  was already fit reuses the stored fit (fitted parameters + traces + pyPESTO result) and
  skips the expensive optimization; the model is still simulated, so the simulation dataframe
  and metrics are always current. The result is otherwise identical, so this is transparent.
  Above that, the fit itself de-duplicates on model *structure*: an unchanged model isn't refit at
  all, and a structure already in the version history points you at `restore_version(n)`.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest + the agent stack
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
