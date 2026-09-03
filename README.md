# petab-mcp

## Overview

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns model
building into a **disciplined, auditable** workflow — the loop behind model-building
assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

- The experimental design (**measurements, conditions, observables**) is read-only. On a
  **blinded** problem (every benchmark example) an observable's formula names a
  `readout_<observableId>` symbol in place of model content, so _what_ was measured stays
  fixed while the mapping onto the agent's own states is the agent's to define — an
  assignment rule, `readout_x := <expression over your species>`.
- The **model is authored and edited only in Antimony**, through one constrained, justified
  operation: **`make_changes` is the single editor**. It takes exactly one of two model
  routes — a batch of **ID-addressed structured operations** (`model_edits`: add/remove a
  reaction, set a rate law, an initial, an assignment rule, rename a symbol; optionally
  applied to a prior version's stored source with `base_version`, which is
  rollback-and-edit in one call) or a **whole-model re-author** (`antimony`, the route for
  the first model and for genuinely starting over) — and/or sets a parameter's
  **scale/bounds**. Never free-form: there is no SBML-input tool.
- Every operation **must cite a justification**; references are **optional** (validated
  against **PubMed** when provided).
- Operations **auto-maintain SBML↔PEtab consistency**: new symbols become model parameters
  or species, orphans are pruned, and a species' **initial condition is parametrised**
  (`S = S_init`) wherever a parameter is the right answer, so the parameter table always
  matches the model. Four cases are deliberately left alone — see the Design notes.
- **Every change is scored.** `make_changes` fits and snapshots a version on every call —
  there is no fit-free edit and no standalone fit, so a change and its score cannot come
  apart. One coherent change is one call.
- Every change is **versioned and logged** (provenance): each version records the fitted
  parameters, fit metrics, and simulation dataframe; alterations and rollbacks are persisted
  to disk with full metadata.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI); optimization uses
[pyPESTO](https://github.com/ICB-DCM/pyPESTO) with the
[Fides](https://github.com/fides-dev/fides) trust-region optimizer. All are **required** —
fitting is not optional machinery.

## Install

```bash
pip install -e .          # installs AMICI + pyPESTO (a C++ toolchain is required)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

## Run

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# options: --provenance-dir DIR   --reference-validation {required,best_effort,off}
#          --build-root DIR       --skills-dir DIR
#          --transport {stdio,sse,streamable-http}
```

The canonical model is Antimony: if a `model.ant` sits next to the YAML it is the source of
truth. A problem may also ship **without** a model (an empty SBML skeleton and no
`model.ant`); it is seeded with a **content-free initial model** — one species per
identifier the fixed observable/condition tables already name, no reactions; a blinded
problem's `readout_*` measurement mappings are seeded as constant **parameters** instead,
since a mapping is not a state and seeding it as one taught agents to wire it into reactions
— so v0 is a valid, simulatable, restorable version rather than a broken empty skeleton, and
`describe_problem` reports `model_authored: false` until the agent zero-shots the mechanism
with `make_changes` (see below). Provenance is written to `<problem_dir>/.petab_mcp/` by
default and is **durable** — restarting resumes the latest version. A ready toy problem
(reversible conversion reaction A⇌B) lives at `tests/data/toy_v1/`.

## Examples

[`examples/`](examples/) holds end-to-end **zero-shot model-building demos**: an
OpenRouter-hosted LLM, driving a panel of three specialist analyst subagents plus a
**blinded biocurator** (via [`deepagents`](https://github.com/langchain-ai/deepagents))
through these tools, is handed a PEtab problem that **ships without a model** (an empty
SBML) and has to **build** the mechanism from the data — one `make_changes(antimony=…)`
call, then refinement. Each has an obvious **trap** (a linear/bilinear/first-order rate law
that can't reproduce the dynamics for any parameters), so fitting the numbers isn't enough;
an independent judge checks the built **mechanism** by BIC against the true model, not just
the fit. Every example also has
a no-LLM `--driver scripted` self-test.

There are **21** examples. Three are synthetic (they simulate their own data from a withheld
truth model, so the trap is planted deliberately):

| Example                                         | System                                    | The trap → the true mechanism                                                        |
| ----------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------ |
| [`enzyme_kinetics/`](examples/enzyme_kinetics/) | Two-reaction enzymatic network, two doses | Michaelis–Menten → **substrate inhibition** (`…/(Km+S+S²/Ki)`)                       |
| [`predator_prey/`](examples/predator_prey/)     | Rosenzweig–MacArthur limit cycle          | Lotka–Volterra bilinear → **Holling type-II** saturating response (2 coupled fluxes) |
| [`repressilator/`](examples/repressilator/)     | Three-gene ring oscillator                | Non-cooperative Hill `n=1` → **cooperative** `n≥2` (3 symmetric reactions)           |

The other eighteen are **real published models** with real experimental data, materialized
from the pinned PEtab Benchmark-Models collection by `python examples/setup.py` (armistead,
blasi, boehm, brannmark, bruno, crauste, fiedler, fujita, oliveira, perelson, rahman,
schwen, sneyd, plus the large networks — 25-36 species, 36-96 reactions, reference k 46-142
— isensee, alkan, bachmann, lucarelli, raimundez, which need a bigger per-task budget: see
`paths.LARGE_EXAMPLES`). Each ships model-less and with **blinded observables**, so the
agent must build both the mechanism and the mapping from its states onto each readout; the
BIC bar is the published model's own fit. See
[`examples/BENCHMARK_NOTES.md`](examples/BENCHMARK_NOTES.md). The roster lives in one place,
`petab_harness.paths` — add a new example there.

See [`examples/README.md`](examples/README.md) for the shared harness (the analyst panel,
Arize Phoenix tracing, judging, and how to run).

## Tools

**Read-only (the fixed design)** — `describe_problem`, `list_observables`,
`list_conditions`, `get_measurements(condition_id?, observable_id?)`, and
`get_data_regimes(observables?)` — what the measurements require of _any_ model, computed
from the data alone (no model, no fit), so it is valid before the first model exists. It is
registered unconditionally; the harness only hands it to the agent under the `data_regimes`
arm, which is **off** by default (measured harmful as built — see
`petab_harness.ablations`).

**The editor** —
`make_changes(justification, antimony?, model_edits?, base_version?, parameters?, references?, n_starts?)`
is the **single** tool that changes anything, and it always fits. There are two model routes
and you give exactly **one**:

- `model_edits` is the route for every change after the first: a **batch** of structured
  operations addressed by ID, not by text — `add_reaction`, `remove_reaction`,
  `set_rate_law`, `set_initial`, `set_assignment` (an `id := expr` rule holding at all
  times, which is how a blinded `readout_*` symbol is defined over your own species),
  `remove_assignment`, `rename`. They apply in order to the model **source** and are
  all-or-nothing: a failure names the failing `model_edits[i]`, lists the valid reaction ids
  when an id is wrong, and applies nothing. Send the whole coherent change in one call — one
  call is one fit and one version.
- `base_version` (with `model_edits`) applies the operations to a **prior version's stored
  source** instead of the current one — the rollback-and-edit path, so an earlier model never
  has to be retyped from memory. It restores that version's search space (bounds and scales)
  and starts the fit from that version's own best fit; a parameter this call names in
  `parameters` keeps your bound rather than the restored one.
- `antimony` authors the **whole** model from an Antimony string: the route for the first
  model (`describe_problem` → `model_authored: false`), for starting over, and for anything
  the operations cannot express. Every reaction needs an id — that is how `model_edits`
  addresses it later. New species/parameters + parametric ICs are reconciled automatically.
- `parameters` batch-edits the bounds and/or scale of one or more parameters: a non-empty
  list of `{parameter_id, lower_bound?, upper_bound?, scale?}` entries (a bound edit gives
  **both** bounds).
- At least one of `antimony` / `model_edits` / `parameters` is required. A model route and
  `parameters` land in the **same transaction**, applied after the reconcile, so an entry may
  target a parameter the change introduces, and one rejected entry reverts the whole call.
- read-only companions: `get_model(format, version?)`, `get_model_structure`.

The `model_edits` + `base_version` arguments are the `diff_edits` arm
(`petab_harness.ablations`), **on by default**; with `PETAB_DIFF_EDITS=0` the schema narrows
to the baseline where `antimony` is the only model route.

Every call then fits and snapshots exactly one version, so a change and its score cannot
come apart. The result **is** the fit — the same field names `get_fit_summary` uses — plus
`applied` (is the change in the model?) and `committed` (did it make a version?). If the
edit applies but the fit fails, the edit is **reverted** (`applied: false` + `error`) —
nothing is left half-done.

**Parameters** — `get_parameters` (read-only; values are set by the fit). Parameter _values_
are set only by the fit; adjust the search space with `make_changes`' `parameters` list.
`get_model_structure` carries the per-species `initial_conditions` account — whether each
declared t=0 is used (fixed), fitted (estimated), or set per-condition.

**The fit** — every `make_changes` call fits the free parameters (pyPESTO multistart),
simulates, and snapshots a single version carrying the fitted parameters, the fit metrics
(`neg_log_likelihood`, `chi2`, `bic`), improvement metrics (`objective_before`,
`self_improvement`, `improvement_vs_best_prior`, `is_new_best` — so versions can be ranked;
`is_new_best`/`improvement_vs_best_prior` track **BIC** against the best model so far, and
`is_new_best` is gated on a minimum effect size, ΔBIC ≥ 2, so noise-floor wander — or an nll
gain that raises BIC — isn't counted as progress), and the simulation dataframe. The result
also carries a `progress` block (non-improving streak, best-BIC `best_version`, best-BIC
gain over the last N ops) that backs an early-stop once the score stops moving — see
`get_progress`. `n_starts` **replaces** the server's configured multistart count
(`create_server(auto_fit_starts=…)`, which the examples harness sets from its own
`--n-starts`; the standalone CLI uses the built-in default), so pass it only to go _above_
that default — a smaller value fits less hard, and the minimum is `1`. Neither the number
nor its effect has to be guessed: `describe_problem` reports the default as
`fit_n_starts_default` and every fit result reports the `n_starts` it actually ran with. The
fit de-duplicates: a model byte-identical to the current version returns the stored metrics
without refitting (`committed: false, no_change: true`), a structure already elsewhere in
the history is fit but **warns** to use `restore_version(n)` instead, and the same structure
fit repeatedly is **refused**. A model that can't compile or has no free parameters returns
`{committed: false, error}` and leaves the working model in place to repair.

**Fit inspection** — `get_fit_summary()` returns just the metrics (no records; a cheap score
check). Two compact, size-bounded helpers give a quick read without pulling the raw data:
`get_fit_diagnosis(sections?)` — `residuals` (per-observable χ²/bias + the worst-fit points;
the default), `parameters` (which estimated parameters are unconstrained or trade off, and
what each one-shared-value merge would cost — the basis for simplifying an
over-parameterised model — a local expansion around the LAST FIT's optimum, read off that
fit's own frozen record and, in a process that did not run it, off the verdict the fit
stored, so **no diagnosis can ever trigger a model build**), or `all` — the cheapest few
merges worth taking ride on every fit result and on `get_fit_summary` as `merge_hints` — and
`get_run_timing()` (cumulative server-side compile/fit wall times). For anything else the
fit is **not** pre-digested into a fixed menu — it is **SQL over the data** via DuckDB, so
you compute the breakdown you want:

- `list_data_tables()` lists the queryable tables and their columns;
  `query_data(sql, max_rows?)` runs a DuckDB query against a **disposable copy** of the
  tables — any SQL is allowed and cannot affect the session (host file/network access is
  disabled) — and returns `{columns, rows, row_count, truncated}`.
- Tables: the fixed design (`measurements`, `conditions`, `observables`), the `parameters`
  table, and the `simulations` dataframe from the last fit — one row per measurement point
  with `measurement`, `simulation`, the raw residual (`residual` =
  `measurement - simulation`, in observable units), the **noise-weighted** residual
  (`residual_weighted` = `(m - s) / σ` on the observable scale — the χ² term, reproducing
  PEtab's `calculate_chi2`), and the evaluated noise σ (`noise`).
- `simulations_history`: those same per-point rows for **every committed version**, plus a
  `version` column — so model versions can be compared point-by-point in one query (which
  residuals got worse between v3 and v4, and where). Assembled from the version store on
  demand; a version whose fit failed stores no simulation and contributes no rows.
- The last fit's **pyPESTO multistart** is queryable too: `optimization_starts` (one row per
  start, `start`=0 is best — `fval`, `time`, `n_fval`/`n_grad`/`n_hess`; the per-start
  `exitflag`/`message` are no longer columns — convergence is reported once per fit as an
  `exitflag_counts` histogram) and `optimization_trace` (the objective `fval` per
  `start`/`iteration`), so convergence and multistart spread are a SQL query away
  (`SELECT count(*), min(fval), max(fval) FROM optimization_starts`). These are **persisted
  per version** (as `optimization.json` plus a pyPESTO `pypesto_result.hdf5` result file,
  cached per model by SBML hash and copied into the version dir), so rolling back to a
  version restores its traces — not just its model. Empty on a simulate-only pass.
- Residual statistics, per-observable/-condition/-timepoint breakdowns, worst-fit points,
  and bias or time-trend checks are all just SQL — filter to `residual_weighted IS NOT NULL`
  for residual stats (a NULL residual, from an unsimulated point or un-evaluable noise
  model, would inflate a `count(*)`), e.g. per-observable χ²
  (`SELECT observable_id, count(*) AS n, sum(residual_weighted*residual_weighted) AS chi2 FROM simulations WHERE residual_weighted IS NOT NULL GROUP BY observable_id ORDER BY chi2 DESC`)
  or the worst points
  (`SELECT * FROM simulations WHERE residual_weighted IS NOT NULL ORDER BY abs(residual_weighted) DESC LIMIT 10`).
  The `simulations` table is generated on demand at cold start (no fit yet).
- The **pyPESTO/Fides optimizer log** is exposed separately by
  `get_fitting_log(tail_lines?)` (and the `petab://fit/log` resource): multistart
  convergence, parameters pinned at bounds, and ill-conditioning — the _numerical_ view of
  the fit, complementing the SQL residual view.

**Provenance, versions & errors** — `get_provenance`, `get_errors`, `list_versions`,
`get_version`, `diff_versions`, `restore_version(version)`, `get_progress`. (The HEAD
version comes from `describe_problem`.)

**Skills** — `list_skills`, `get_skill(name, file?)`. The server ships tool-usage
documentation as [agent skills](https://modelcontextprotocol.io) bundled under
`src/petab_mcp/_skills/` and exposes them over MCP so a client can learn the workflow
without leaving the session (see below).

Resources: `petab://model/antimony`, `petab://model/sbml`, `petab://observables`,
`petab://conditions`, `petab://measurements`, `petab://parameters`, `petab://provenance`,
`petab://versions`, `petab://errors`, `petab://simulations`, `petab://data/schema`,
`petab://fit/log`, `petab://skills`, `petab://skills/{name}`.

## Skills

`src/petab_mcp/_skills/` holds self-describing **skills** — markdown docs that explain how
to use this server's tools — in the standard agent-skill layout (a directory per skill with
a `SKILL.md` frontmatter header plus supporting files). Bundling them inside the package
means they resolve identically from a source checkout and an installed wheel. Three ship:
`petab-model-building` documents the disciplined edit→fit→audit workflow,
`antimony-authoring` covers writing the model text `make_changes` takes, and
`independence-audit` covers what to do when residuals stay structured after re-fitting.

Skills are exposed to MCP clients as:

- tools — `list_skills()` and `get_skill(name, file?)` (the reliable path, since every
  client consumes tools);
- resources — `petab://skills` (a JSON index) and `petab://skills/{name}` (a skill's
  `SKILL.md`).

Override the directory with `--skills-dir DIR` or `PETAB_MCP_SKILLS_DIR` to point at your
own skills; drop another skill directory into `_skills/` to extend the bundled docs.

## Design notes

- **Constrained, justified editing.** There is no free-form SBML-input tool. Each edit runs
  one pipeline: validate references → apply to a model copy → reconcile SBML↔PEtab → lint →
  apply or revert → log. An edit that breaks a fixed observable/condition is rejected, and
  because the revert covers the model _and_ the parameter table, a call that carries both a
  model change and a `parameters` list is atomic across both. A `model_edits` batch is
  all-or-nothing in the same way: operations apply in order to the model _source_ (the text
  the agent last authored or edited, never the reconcile's re-render), and a failure names
  the failing `model_edits[i]` and applies nothing.
- **Parametric initial conditions.** `reconcile` parametrises a species' initial condition
  as an `initialAssignment(S := S_init)` with `S_init` in the parameter table, so an initial
  condition is a first-class parameter, and it is **estimated by default** — a declared t=0
  is only a guess and must never be silently frozen (a fixed guess the optimizer can't
  correct is the trap behind #90). An IC of exactly `1` is the leak-free exception: a unit
  normalization (e.g. a conserved-total set to 1) is held **fixed by value**, so it names no
  species. To keep any other IC fixed, ship it as a fixed parameter row — reconcile never
  re-touches existing rows. It also mints **nothing** in four cases, each because minting
  would be wrong rather than merely unnecessary: a species initialised to exactly `0` or
  left unset (an empty pool is the author's design, and a zero nominal breaks log-scale
  estimation, so the literal stays and no row appears), a species whose initial condition is driven by a
  **condition-table column** per condition (the override would silently supersede an
  estimated row), a species targeted by an **assignment rule** (minting an
  `initialAssignment` alongside it builds SBML that libSBML rejects — the server would be
  rejecting the agent's own valid model), and an author's own **compound** parametric IC
  such as `Rec2 = ini_R1*ini_R2fold` (its parameters already _are_ the estimable IC).
  `get_model_structure`'s `initial_conditions` block reports each species' actual mode.
- **PubMed validation.** References are **optional** on every operation (only a non-empty
  `justification` is required); when given, they are resolved via NCBI E-utilities (no extra
  dependency). The mode governs how strictly a _supplied_ reference is validated: `required`
  rejects one that cannot be resolved; `best_effort` (default) records a `verified` flag and
  never blocks on the network; `off` skips validation.
- **Provenance, versioning & error logging.** `.petab_mcp/provenance.jsonl` is an
  append-only event log; `.petab_mcp/versions/vN/` snapshots the Antimony model + parameter
  table (as a PEtab `parameters.tsv`, not JSON) plus `model_source.ant` — the text the agent
  itself submitted, which is what a `base_version` fork and `get_model(version=N)` read back
  — and, for a fitted version, its `simulations.json`, `optimization.json` (pyPESTO traces),
  and `pypesto_result.hdf5` (the pyPESTO result file, staged in `.petab_mcp/cache/` and
  copied in); `.petab_mcp/errors.jsonl` records rejected operations /
  failed tool calls (surfaced via `get_errors`); `.petab_mcp/petab_mcp.log` is a
  human-readable log (including a concise per-fit summary), and `.petab_mcp/pypesto.log`
  captures pyPESTO's own multistart optimizer report for each fit. `restore_version`
  restores a prior version as a new, auditable version — including its fit artifacts
  (simulation + optimization traces + result file), copied forward. A version is exactly a
  fit snapshot: a call appends an `edit` event and the fit it runs creates the version that
  scores it.
- **One version per coherent change + comparison.** Each fit records its objective and how
  it compares to the best prior version, so competing model versions can be ranked by fit
  without a separate optimize step — and because a whole change (a model edit or re-author
  plus the bounds it needs) is one call, it is scored once instead of appending a version
  per micro-edit.
- **Cheap parameter edits, cached compilation.** AMICI models are cached by SBML hash; a
  structural change recompiles, parameter edits do not.
- **Warm starts.** Half of every multistart (`WARM_START_FRACTION`) is seeded from the
  **previous** fit rather than sampled: its converged optima, best-first, preceded by rows
  that put each parameter the edit just introduced at a _neutral_ value (unit scale, then
  its lower bound, then its upper bound — a gate neutralises low, a pass-through rate high,
  and which one applies cannot be inferred). So an additive edit's first start sits where
  the child reproduces its parent, which is what makes a nested comparison meaningful; the
  remaining starts are sampled fresh. Deterministic — the jitter uses a private RNG seeded
  from `FIT_SEED`, so it neither depends on nor perturbs pyPESTO's own stream. A fit
  therefore depends on the run's history, not on the estimation problem alone: there is no
  fit result cache (one existed and was removed with the decoupled-edit machinery).
- **Structure de-duplication.** Instead of caching fits, the server refuses to spend one
  twice: a model byte-identical to the current version returns the stored metrics without
  refitting (`committed: false, no_change: true`), a structure already elsewhere in the
  version history is fit but **warns** to use `restore_version(n)`, and the same structure
  fit repeatedly is **refused**. Identity is the model structure plus the estimation setup —
  bounds, scale, which parameters are estimated, fixed values
  (`dedup.estimation_setup_signature`).
- **Invalid comparisons are flagged, not silently ranked.** A commit whose
  estimated-parameter count rose while its nll _also_ rose cannot be nested in the parent it
  is compared against — either the edit displaced an existing term instead of extending it,
  or the optimizer under-converged — so the payload says so rather than letting BIC report
  the extra structure as unhelpful (`nesting`). Alongside it, `get_progress` carries the
  early-stop chain (non-improving streak, best BIC, goodness-of-fit p-value,
  `should_stop`/`stop_reason`), so a run stops when the score stops moving instead of burning
  fits on a plateau.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest + the agent stack
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
