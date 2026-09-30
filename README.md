# gemot

[![CI](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The agent harness built around [PEtab MCP](https://github.com/frohlich-lab/PEtab-MCP), an MCP
server that wraps a single [PEtab v1](https://petab.readthedocs.io) parameter-estimation problem
and lets an LLM agent change the model only through one constrained tool, with every change
**scored** (fit and simulated), **versioned** and **justified**.

gemot spawns that server as a subprocess, launches a panel of agent specialists (mathematician,
systems scientist, signal engineer) on it and then hands off to the **judge**, which selects the
best-scoring version and scores it against a true or published reference. Phoenix tracing and the
run explorer read the trace and version history this pipeline leaves behind.

The benchmark it drives — the zero-shot examples, the Slurm sweep tooling and the paper — lives in
[frohlich-lab/petab_mcp](https://github.com/frohlich-lab/petab_mcp), which installs gemot pinned
to a commit. gemot is a checkout tool: it finds that checkout from the running script or the
working directory, or from `PETAB_MCP_REPO_ROOT` (see [`src/gemot/paths.py`](src/gemot/paths.py)).

## Install

```bash
pip install -e .          # gemot + the pinned server (AMICI + pyPESTO; a C++ toolchain is required)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

The server repository is private, and pip fetches it with `git`, so `git` needs read access to
it — through `gh auth setup-git`, or by fetching that one URL over SSH:

```bash
git config --global url."git@github.com:frohlich-lab/PEtab-MCP".insteadOf https://github.com/frohlich-lab/PEtab-MCP
```

To work on the server and gemot together, install your server checkout over the pinned one
(and again after any later `pip install -e .`, which restores the pin):

```bash
pip install -e ../PEtab-MCP
```

## Project structure

[`gemot`](src/gemot/) (under `src/`); each submodule is documented in its own docstrings.

- `gemot.runner` — cleanup → server → driver → judge, and the command line every example's
  `run.py` drives.
- `gemot.agent` — the deepagents panel: tool curation, guards, finish gate, driver.
  `--driver barebones` is the **null arm**: the same model, problem, libraries and compute, but
  a stock deepagent with file tools and a sandboxed shell over the PEtab files on disk — no
  server, no panel, no scored editing loop — asked to deliver a finished PEtab problem in place,
  which the same judge then scores at its delivered parameter values (see
  [`src/gemot/agent/barebones.py`](src/gemot/agent/barebones.py)).
- `gemot.neural_ode` — the non-LLM model producer. It reads a PEtab problem's tables without
  opening the shipped model, restates the problem as a PEtab SciML neural ODE, fits it through
  AMICI's JAX backend, scores it against the example's held-out set, and distils the field into
  Antimony rate laws with PySR for the same judge. `python -m gemot.neural_ode fit|propose` runs
  either half.
- `gemot.explorer` — the run viewer over archived and in-flight runs.
- `gemot.analysis`, `gemot.ablations` — sweep analysis (`python -m gemot.analysis`) and the
  registry of every switch that changes what a run does (`python -m gemot.ablations`).
- `gemot.paper*` — the manuscript's generated inputs and figures, built from the benchmark
  checkout's `paper/`.

## Design notes

The principles gemot is built on are the server's — the single scored editing tool, the one
named score every gate reads, the versioned provenance. They are stated in the
[PEtab MCP README](https://github.com/frohlich-lab/PEtab-MCP#design-notes).

## Development

```bash
pip install -e '.[dev]'   # dev adds pytest + the agent stack
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
