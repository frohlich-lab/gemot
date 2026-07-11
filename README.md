# petab-mcp

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns model
building into a **disciplined, auditable** workflow — the loop behind model-building
assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

- The experimental design (**measurements, conditions, observables**) is read-only.
- The **model is authored and edited only in Antimony**, and only through three
  constrained operations: **add a reaction, change a rate law, remove a reaction**.
- Every operation **must cite a justification and ≥1 reference**, validated against
  **PubMed**.
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
source of truth. Provenance is written to `<problem_dir>/.petab_mcp/` by default and is
**durable** — restarting resumes the latest version. A ready toy problem (reversible
conversion reaction A⇌B) lives at `tests/data/toy_v1/`.

## Tools

**Read-only (the fixed design)** — `describe_problem`, `list_observables`,
`list_conditions`, `get_measurements(condition_id?, observable_id?)`.

**Model operations** (each requires `justification`; model ops also require `references`) —
- `add_reaction(reaction, justification, references)` — `reaction` is an Antimony line,
  e.g. `"J: A + B -> C; k_f*A*B"`; new species/parameters + parametric ICs are added
  automatically.
- `change_rate_law(reaction_id, rate_law, justification, references)`
- `remove_reaction(reaction_id, justification, references)` — prunes orphaned entities.
- `change_parameter_scale(parameter_id, scale, justification, references?)` — references optional.
- `change_parameter_bounds(parameter_id, lower_bound, upper_bound, justification, references?)`.
- read-only: `get_model(format)`, `get_model_structure`, `validate_problem`.

**Batching** — `begin_batch(label)` → run several operations (applied + logged, not yet
versioned) → `commit_batch(label)` snapshots them as **one** version; `abort_batch`
discards staged edits. `optimize`/`rollback_to`/`reset` are blocked while a batch is open.

**Parameters** — `get_parameters` (read-only; values are set by the fit), `sync_parameters`.
Parameter *values* are set only by the automatic fit; adjust the search space with the
versioned `change_parameter_bounds` / `change_parameter_scale` operations.

**Fit + simulation (automatic on commit)** — every commit fits the free parameters
(pyPESTO multistart) and simulates, then snapshots a single version carrying the fitted
parameters, the fit metrics (`neg_log_likelihood`, `chi2`), improvement metrics
(`objective_before`, `self_improvement`, `improvement_vs_best_prior`, `is_new_best` — so
versions can be ranked by objective), and the simulation dataframe. Pass `n_starts` on a
commit to fit harder (`0` = simulate without fitting); batching fits once on `commit_batch`.

**Fit inspection** — `get_simulations()` returns the stored simulation dataframe, and
`inspect_fit(top_n)` breaks the fit down beyond the scalar objective (both read the stored
dataframe — no re-simulation):
- Per-point records carry the raw residual (`measurement - simulation`, in observable
  units), the **noise-weighted** residual (`(m - s) / σ` on the observable scale — the
  term that enters χ²), and the evaluated noise σ. Weighted residuals reproduce PEtab's
  own `calculate_chi2` exactly.
- `inspect_fit` aggregates residual statistics **overall** and across each cross section —
  **observables**, **experiments/conditions**, **timepoints** — reporting counts,
  mean/std/RMSE/SSE of raw residuals, and weighted stats (`chi2`, `reduced_chi2`,
  `mean_weighted_residual`, and a bias z-score), plus the `top_n` **worst-fit points** and
  **bias flags** for groups with a systematic over/under-prediction.

**Provenance, versions & errors** — `get_provenance`, `get_errors`, `list_versions`,
`get_version`, `diff_versions`, `rollback_to(version)`, `current_version`, `reset`.

**Skills** — `list_skills`, `get_skill(name, file?)`. The server ships tool-usage
documentation as [agent skills](https://modelcontextprotocol.io) bundled under
`src/petab_mcp/_skills/` and exposes them over MCP so a client can learn the workflow
without leaving the session (see below).

Resources: `petab://model/antimony`, `petab://model/sbml`, `petab://observables`,
`petab://conditions`, `petab://measurements`, `petab://parameters`,
`petab://provenance`, `petab://versions`, `petab://errors`, `petab://fit`,
`petab://simulations`, `petab://skills`, `petab://skills/{name}`.

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
  `initialAssignment(S := S_init)` with `S_init` in the parameter table (linear scale),
  so initial conditions are first-class, estimatable parameters.
- **PubMed validation.** References are resolved via NCBI E-utilities (no extra
  dependency). `required` rejects unresolvable references; `best_effort` (default) records
  a `verified` flag and never blocks on the network; `off` skips validation. References
  are always required.
- **Provenance, versioning & error logging.** `.petab_mcp/provenance.jsonl` is an
  append-only event log; `.petab_mcp/versions/vN/` snapshots the Antimony model +
  parameter table; `.petab_mcp/errors.jsonl` records rejected operations / failed tool
  calls (surfaced via `get_errors`); `.petab_mcp/petab_mcp.log` is a human-readable log
  (including a concise per-commit fit summary), and `.petab_mcp/pypesto.log` captures
  pyPESTO's own multistart optimizer report for each fit.
  `rollback_to` restores a prior version as a new, auditable version. Batching chains
  several operations into a single version.
- **Fit on commit + comparison.** Every commit fits and records its objective and how it
  compares to the best prior version, so competing model versions can be ranked by fit
  without a separate optimize step.
- **Cheap parameter edits, cached compilation.** AMICI models are cached by SBML hash;
  a structural operation recompiles, parameter edits do not.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest
pytest                    # AMICI/pyPESTO fit tests skip if absent; PubMed path is stubbed
```

## License

MIT
