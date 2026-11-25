# gemot

[![CI](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The agent harness built around [PEtab MCP](https://github.com/frohlich-lab/PEtab-MCP), an
MCP server that wraps a single [PEtab v1](https://petab.readthedocs.io) parameter-estimation
problem and lets an LLM agent change the model only through one constrained tool, with every
change **scored** (fit and simulated), **versioned** and **justified**.

gemot spawns that server as a subprocess, launches a panel of agent specialists
(mathematician, systems scientist, signal engineer) on it and then hands off to the
**judge**, which selects the best-scoring version and scores it against a true or published
reference — or, for a model-free data package with no reference, by an absolute goodness-of-fit
test. Phoenix tracing and the run explorer read the trace and version history this pipeline
leaves behind.

The zero-shot problems it drives — and the code that materializes them from the PEtab
Benchmark-Models collection and fits their reference bars — live in
[frohlich-lab/PEtabAgentBench](https://github.com/frohlich-lab/PEtabAgentBench). The Slurm sweep
tooling and the paper live in
[frohlich-lab/gemot_iclr_2026](https://github.com/frohlich-lab/gemot_iclr_2026), which mounts
those problems at `examples/` and installs gemot pinned to a commit. gemot is a checkout tool: it
finds that checkout
from the running script or the working directory, or from `PETAB_MCP_REPO_ROOT` (see
[`src/gemot/paths.py`](src/gemot/paths.py)).

## Quickstart

### Installation and dependencies

Python 3.12+ and a C++ toolchain for AMICI (see [Troubleshooting](#troubleshooting)).

```bash
pip install -e .              # gemot + the pinned server release from PyPI
pip install -e '.[harness]'   # + the deepagents panel and Phoenix tracing, for LLM-driven runs
```

To work on the server and gemot together, install your server checkout over the pinned one
(and again after any later `pip install -e .`, which restores the pin):

```bash
pip install -e ../PEtab-MCP
```

### Running an example

The problems come from [PEtabAgentBench](https://github.com/frohlich-lab/PEtabAgentBench).
Check it out inside this repo at `examples/` so gemot finds it, then let the deepagents
panel build the synthetic predator–prey model through an OpenRouter-hosted model:

```bash
git clone https://github.com/frohlich-lab/PEtabAgentBench.git examples
export OPENROUTER_API_KEY=sk-or-...         # go to https://openrouter.ai/keys to get your key
python examples/predator_prey/run.py        # --driver deepagents (default)
```

The full example can take at least an hour to run on a laptop, so you may want to track the
agent's progress via the Phoenix trace. To view this, serve Phoenix in a second terminal and
point the run at it:

```bash
phoenix serve                                                   # UI at http://localhost:6006
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006         # in the run's terminal
```

The run exits `0` if the judge accepts the built model, writes its verdict to
`examples/predator_prey/work/result.json` and archives the run under
`examples/_runs_archive/`. The archive keeps each version's model, parameters, statements,
simulations, multistart (`optimization.json.gz`) and pyPESTO's HDF5 result.

`--model` picks another OpenRouter model, and `--driver scripted` builds the true model
without an LLM as a self-test of the pipeline.

### Codex and Claude CLI drivers

Run a native CLI agent for the whole task with `.[harness]` installed and the CLI authenticated:

```bash
codex login
python examples/predator_prey/run.py --driver codex --model YOUR_CODEX_MODEL_ID
claude auth login
python examples/predator_prey/run.py --driver claude --model sonnet
```

Each run starts one `codex exec` or `claude -p` session and supplies the task once. The CLI
owns its conversation, tool loop, prompt caching, and compaction. These drivers have no gemot
panel or finish gate. They use the shared scored PEtab MCP tools through a local gateway;
file/shell access and heldout judging tools are restricted to preserve benchmark blindness.

`--model` is required and passed directly to the CLI. No OpenRouter key is needed.
`--biocurator-model` is unavailable; `--recursion-limit` does not apply. Optional
`--max-wall-seconds` or `--codex-timeout` / `--claude-timeout` bounds the entire session.

Phoenix records observed native events and actual gateway tool calls under the run trace.
Codex usage is session aggregate; Claude additionally exposes response usage. Raw events are
archived as `work/codex_events.jsonl` or `work/claude_events.jsonl`. See
[CLI drivers](docs/CLI_DRIVERS.md) for setup, trace limits, and the ChatGPT terms assessment.

## Project structure

[`gemot`](src/gemot/) (under `src/`); each submodule is documented in its own docstrings.

- `gemot.runner` — cleanup → server → driver → judge, and the command line every example's
  `run.py` drives.
- `gemot.agent` — the deepagents panel: tool curation, guards, finish gate, driver.
  `--driver barebones` is the **null arm**: the same model, problem, libraries and compute,
  but a stock deepagent with file tools and a sandboxed shell over the PEtab files on disk —
  no server, no panel, no scored editing loop — asked to deliver a finished PEtab problem in
  place, which the same judge then scores at its delivered parameter values (see
  [`src/gemot/agent/barebones.py`](src/gemot/agent/barebones.py)).
- `gemot.neural_ode` — the non-LLM model producer. It reads a PEtab problem's tables without
  opening the shipped model, restates the problem as a PEtab SciML neural ODE, fits it
  through AMICI's JAX backend, scores it against the example's held-out set, and distils the
  field into Antimony rate laws with PySR for the same judge.
  `python -m gemot.neural_ode fit|propose` runs either half.
- `gemot.explorer` — the run viewer over archived and in-flight runs.
- `gemot.analysis`, `gemot.ablations` — sweep analysis (`python -m gemot.analysis`) and the
  registry of every switch that changes what a run does (`python -m gemot.ablations`).
- `gemot.causal_constraints` — what the `causal_constraints` arm records: the refusals per
  constraint, the first version making them all, and how the judged model behaves on each
  indirect constraint, read with the gain norms of Fröhlich et al., MSB 2023.

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

## Troubleshooting

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

