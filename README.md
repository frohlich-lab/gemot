# petab-mcp

[![CI](https://github.com/frohlich-lab/petab_mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/petab_mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns mechanistic
model building into a **disciplined, auditable** workflow for LLM agents — the loop behind
model-building assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

Point an agent at a dataset and it proposes and refines a mechanistic model, one justified
change at a time. The experimental design stays fixed and the model is edited only through
one constrained tool, so an agent cannot silently rewrite the rules of the problem. Every
change is **scored** on the spot (fit and simulated), **versioned** (nothing overwritten,
nothing lost), and **justified** (a rationale attached, optionally checked against PubMed).
This means that instead of a black box just delivering a model, you get the model and
inspectable provenance describing how it developed.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI), optimization uses
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

There are three entry points.

**1. The MCP server**, standalone, against any PEtab problem:

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# --provenance-dir DIR   --reference-validation {required,best_effort,off}
# --build-root DIR       --skills-dir DIR
# --transport {stdio,sse,streamable-http}
```

Provenance is written to `<problem_dir>/.petab_mcp/` by default and is durable — restarting
resumes the latest version. A ready toy problem lives at `tests/data/toy_v1/`.

**2. A zero-shot model-building example**, driving the agent harness against the server:

```bash
python examples/enzyme_kinetics/run.py                     # --driver deepagents (default)
python examples/enzyme_kinetics/run.py --driver scripted   # no LLM, self-test
```

**3. The run explorer**, to browse archived and in-flight runs:

```bash
python examples/explore_ui.py
```

Examples also run as Slurm array jobs (env setup, a shared Phoenix collector, sweeps) — see
[`slurm/README.md`](slurm/README.md).

## Project structure

Two packages under `src/`: [`petab_mcp`](src/petab_mcp/) (the MCP server) and
[`petab_harness`](src/petab_harness/) (the agent harness that drives it — panel, judge,
benchmark sweep, run explorer). Each submodule is documented in its own docstrings.

### Examples

[`examples/`](examples/) holds 21 zero-shot model-building demos: a PEtab problem that ships
without a model, which an LLM agent panel must build from the data alone. Each has a
deliberate trap, so fitting the numbers isn't enough — an independent judge scores the built
mechanism by BIC against a true or published reference. Three are synthetic
(`enzyme_kinetics/`, `predator_prey/`, `repressilator/`); eighteen are real published models
materialized from the PEtab Benchmark-Models collection (`python examples/setup.py`). See
[`examples/README.md`](examples/README.md) (harness, tracing, judging) and
[`examples/BENCHMARK_NOTES.md`](examples/BENCHMARK_NOTES.md) (the model roster).

### Tools

`make_changes` is the single tool that changes the model or its parameter search space, and
it always fits — see its docstring in [`src/petab_mcp/server.py`](src/petab_mcp/server.py)
for the full argument and result reference. Everything else is read-only: problem/model/
parameter inspection (`describe_problem`, `get_model`, `get_parameters`, ...), fit
inspection (`get_fit_summary`, `get_fit_diagnosis`, `query_data` — arbitrary SQL over the
fit via DuckDB), and provenance (`get_provenance`, `list_versions`, `restore_version`, ...).
A live server also exposes these as MCP resources under `petab://...`.

### Harness

The [`petab_harness`](src/petab_harness) drives a model building example. It spawns the MCP
server as a subprocess, launches a panel of agent specialists (mathematician, systems
scientist, signal engineer) and then hands off to the **judge**, which selects the
best-scoring version and scores it against a true or published reference. Phoenix tracing
and the run explorer read the trace and version history this pipeline leaves behind.

## Design notes

Principles to hold onto when extending this package.

- **No free-form editing.** The model is only ever changed through `make_changes`, and every
  change must carry a justification. Don't add a second way to write to the model.
- **A change and its score never come apart.** `make_changes` always fits and snapshots a
  version — there is no edit without a fit, and no fit without a version.
- **SBML and PEtab stay consistent automatically.** Reconciliation (new symbols, orphan
  pruning, parametric initial conditions) is the server's job, not the agent's — but a few
  cases (an IC of exactly 0, a condition-table override, an assignment rule, a compound
  expression) are deliberately left untouched because minting there would be wrong, not just
  unnecessary. Read [`src/petab_mcp/reconcile.py`](src/petab_mcp/reconcile.py) before
  "fixing" one of these.
- **Every operation is versioned and logged.** Provenance under `.petab_mcp/` is append-only
  and durable; extend what a version records rather than bypassing it.
- **Iteration is cheap; results are not cached.** Structural edits recompile, parameter
  edits don't — but an identical or already-seen structure is refused a wasted refit, never
  served from a fit-result cache (there isn't one).
- **Fit inspection is SQL, not a growing menu.** A new way to look at a fit should be a
  `query_data` query over the DuckDB tables, not a new bespoke tool.
- **Comparisons must be valid before they're reported.** A commit that isn't nested in what
  it's compared against is flagged, not silently ranked by BIC.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest + the agent stack
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
