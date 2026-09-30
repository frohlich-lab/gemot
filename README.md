# petab-mcp harness

[![CI](https://github.com/frohlich-lab/petab_mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/petab_mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The agent harness, zero-shot benchmark and paper pipeline built around
[PEtab MCP](https://github.com/frohlich-lab/PEtab-MCP): an MCP server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and lets an LLM agent
change the model only through one constrained tool, with every change **scored** (fit and
simulated), **versioned** and **justified**. The server lives in its own repository and is
installed from there, pinned to a commit in [`pyproject.toml`](pyproject.toml).

## Install

```bash
pip install -e .          # the harness + the pinned server (AMICI + pyPESTO; a C++ toolchain is required)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

The server repository is private, and pip fetches it with `git`, so `git` needs read access to
it — through `gh auth setup-git`, or by fetching that one URL over SSH:

```bash
git config --global url."git@github.com:frohlich-lab/PEtab-MCP".insteadOf https://github.com/frohlich-lab/PEtab-MCP
```

To work on the server and the harness together, install your server checkout over the pinned
one (and again after any later `pip install -e .`, which restores the pin):

```bash
pip install -e ../PEtab-MCP
```

## Run

There are two entry points (the server's own CLI, `petab-mcp`, comes with the install).

**1. A zero-shot model-building example**, driving the agent harness against the server:

```bash
python examples/enzyme_kinetics/run.py                     # --driver deepagents (default)
python examples/enzyme_kinetics/run.py --driver scripted   # no LLM, self-test
```

**2. The run explorer**, to browse archived and in-flight runs:

```bash
python examples/explore_ui.py
```

Examples also run as Slurm array jobs (env setup, a shared Phoenix collector, sweeps) — see
[`slurm/README.md`](slurm/README.md).

## Project structure

[`petab_harness`](src/petab_harness/) (under `src/`) is the agent harness that drives the
server — panel, judge, benchmark sweep, run explorer. Each submodule is documented in its own
docstrings.

### Examples

[`examples/`](examples/) holds 21 zero-shot model-building demos: a PEtab problem that ships
without a model, which an LLM agent panel must build from the data alone. Each has a
deliberate trap, so fitting the numbers isn't enough — an independent judge scores the built
mechanism against a true or published reference under the run's own complexity criterion
(BIC by default). Three are synthetic
(`enzyme_kinetics/`, `predator_prey/`, `repressilator/`); eighteen are real published models
materialized from the PEtab Benchmark-Models collection (`python examples/setup.py`). See
[`examples/README.md`](examples/README.md) (harness, tracing, judging) and
[`examples/BENCHMARK_NOTES.md`](examples/BENCHMARK_NOTES.md) (the model roster).

### Harness

The [`petab_harness`](src/petab_harness) drives a model building example. It spawns the MCP
server as a subprocess, launches a panel of agent specialists (mathematician, systems
scientist, signal engineer) and then hands off to the **judge**, which selects the
best-scoring version and scores it against a true or published reference. Phoenix tracing
and the run explorer read the trace and version history this pipeline leaves behind.

`--driver barebones` is the **null arm** for that design: the same model, problem, libraries
and compute, but a stock deepagent with file tools and a sandboxed shell over the PEtab files
on disk — no server, no panel, no scored editing loop — asked to deliver a finished PEtab
problem in place, which the same judge then scores at its delivered parameter values. What
the harness adds is the difference between the two arms on the same sweep (see
[`src/petab_harness/agent/barebones.py`](src/petab_harness/agent/barebones.py)).

[`petab_harness.neural_ode`](src/petab_harness/neural_ode) is the harness's non-LLM model
producer. It reads a PEtab problem's tables without opening the shipped model, restates the
problem as a PEtab SciML neural ODE — the observed and dosed species as states, a tanh network as
the right-hand side, the shipped observation model kept — fits it through AMICI's JAX backend,
scores it against the example's held-out set, and distils the field into Antimony rate laws with
PySR for the same judge. `python -m petab_harness.neural_ode fit|propose` runs either half.

## Design notes

The principles the harness is built on are the server's — the single scored editing tool, the
one named score every gate reads, the versioned provenance. They are stated in the
[PEtab MCP README](https://github.com/frohlich-lab/PEtab-MCP#design-notes).

## Development

```bash
pip install -e '.[dev]'   # dev adds pytest + the agent stack
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
