# gemot

[![CI](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/gemot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gemot.svg)](https://pypi.org/project/gemot/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/frohlich-lab/gemot/blob/main/LICENSE)

## Disclaimer

Research software under active, rapid and heavy development: **use it at your own risk**. Its
command line, result fields and agent prompts change without notice. A run calls a hosted LLM
for hours and is billed by that provider. Its behaviour has been tuned and checked on benchmark
problems only, and it is not for clinical or regulatory use. It comes without warranty of any
kind; see the [LICENSE](https://github.com/frohlich-lab/gemot/blob/main/LICENSE).

## Overview

The agent harness built around [PEtab MCP](https://github.com/frohlich-lab/PEtab-MCP), an
MCP server that wraps a single
[PEtab v2](https://petab.readthedocs.io/en/latest/v2/documentation_data_format.html)
parameter-estimation problem and lets an LLM agent change the model only through one
constrained tool, with every change **scored** (fit and simulated), **versioned** and
**justified**. The preprint
[GEMOT: Towards Mechanistic World Models for Biology](https://www.biorxiv.org/content/10.64898/2026.09.26.754619v1)
describes the framework and its evaluation on 18 published biological problems.

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
those problems at `examples/` and installs gemot pinned to a commit. gemot finds that checkout
from the running script or the working directory, or from `PETAB_MCP_REPO_ROOT` (see
[`src/gemot/paths.py`](https://github.com/frohlich-lab/gemot/blob/main/src/gemot/paths.py)).

## Quickstart

### Installation and dependencies

Python 3.12+ and a C++ toolchain for AMICI (see [Troubleshooting](#troubleshooting)).

```bash
pip install gemot              # gemot + the pinned server release
pip install "gemot[harness]"   # + the deepagents panel and Phoenix tracing, for LLM-driven runs
```

The `indra_statements` arm builds models from INDRA statements and needs the `statements`
extra. No released pysb installs beside AMICI, so name its commit:

```bash
pip install "gemot[harness,statements]" "pysb @ git+https://github.com/pysb/pysb.git@9fc6bcb118683ad851e0a4d5604abe646f77cae3"
```

From a checkout, the `pins` group (pip >= 25.1) adds the unreleased commits PyPI refuses as
dependencies: pysb's, the neural route's petab-sciml, and petab's, which imports a problem of
thousands of experiments (Froehlich_CellSystems2018):

```bash
pip install -e '.[harness,statements]' --group pins
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
[CLI drivers](https://github.com/frohlich-lab/gemot/blob/main/docs/CLI_DRIVERS.md) for setup, trace limits, and the ChatGPT terms assessment.

## Project structure

[`gemot`](https://github.com/frohlich-lab/gemot/tree/main/src/gemot) (under `src/`); each
submodule is documented in its own docstrings.

- `gemot.runner` — cleanup → server → driver → judge, and the command line every example's
  `run.py` drives.
- `gemot.agent` — the deepagents panel: tool curation, guards, finish gate, driver.
  `--driver barebones` is the **null arm**: the same model, problem, libraries and compute,
  but a stock deepagent with file tools and a sandboxed shell over the PEtab files on disk —
  no server, no panel, no scored editing loop — asked to deliver a finished PEtab problem in
  place, which the same judge then scores at its delivered parameter values (see
  [`src/gemot/agent/barebones.py`](https://github.com/frohlich-lab/gemot/blob/main/src/gemot/agent/barebones.py)).
- `gemot.neural_ode` — the non-LLM model producer. It reads a PEtab problem's tables without
  opening the shipped model, restates the problem as a PEtab SciML neural ODE, fits it
  through AMICI's JAX backend, scores it against the example's held-out set, and distils the
  field into Antimony rate laws with PySR for the same judge.
  `python -m gemot.neural_ode fit|propose` runs either half.
- `gemot.explorer` — archived and in-flight runs in petab-mcp's viewer of their version store,
  (`python -m gemot.explorer`, the `explorer` extra).
- `gemot.sweep` — the Slurm sweep driver for a checkout that pins gemot: one job array of runs
  whose fits share one pool of SLURM workers, and the pool, trace collector and explorer as
  service jobs (`python -m gemot.sweep`, [docs/SWEEP.md](https://github.com/frohlich-lab/gemot/blob/main/docs/SWEEP.md)).
- `gemot.analysis`, `gemot.ablations` — sweep analysis (`python -m gemot.analysis`) and the
  registry of every switch that changes what a run does (`python -m gemot.ablations`).
- `gemot.causal_constraints` — what the `causal_constraints` arm records: the refusals per
  constraint, the first version making them all, and how the judged model behaves on each
  indirect constraint, read with the gain norms of Fröhlich et al., MSB 2023.

## Design notes

The principles gemot is built on are the server's — the single scored editing tool, the one
named score every gate reads, the versioned provenance. They are stated in the server's
[design notes](https://github.com/frohlich-lab/PEtab-MCP/blob/main/docs/DESIGN.md).

## Development

```bash
pip install -e '.[dev]' --group pins   # dev adds pytest, the agent stack and every extra
pytest -n 3                            # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

A release is cut by publishing a GitHub release tagged `v<version>`, which the
[release workflow](https://github.com/frohlich-lab/gemot/blob/main/.github/workflows/release.yml)
uploads to PyPI. The version is `[project] version` in `pyproject.toml`; bump `version` and
`date-released` in `CITATION.cff` with it.

## Citation

gemot is described in the preprint
[GEMOT: Towards Mechanistic World Models for Biology](https://www.biorxiv.org/content/10.64898/2026.09.26.754619v1)
(de Pomereu, Snelling & Fröhlich, bioRxiv 2026). If you use gemot, please cite it:

```bibtex
@article{dePomereu2026gemot,
  author  = {de Pomereu, Theodore and Snelling, Branwen and Fr{\"o}hlich, Fabian},
  title   = {{GEMOT}: Towards Mechanistic World Models for Biology},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.26.754619},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.26.754619v1}
}
```

[`CITATION.cff`](https://github.com/frohlich-lab/gemot/blob/main/CITATION.cff) holds the same
reference, which GitHub's "Cite this repository" button (in the sidebar) renders as APA or BibTeX.

Please also cite the server it drives, [PEtab MCP](https://github.com/frohlich-lab/PEtab-MCP), and
the simulation and optimization tools underneath it,
[AMICI](https://github.com/AMICI-dev/AMICI) and [pyPESTO](https://github.com/ICB-DCM/pyPESTO).

## License

MIT; see [LICENSE](https://github.com/frohlich-lab/gemot/blob/main/LICENSE).

## Troubleshooting

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH`. AMICI ships only a source distribution, so it is built on install
(BLAS comes from the `scipy-openblas64` wheel, or Accelerate on macOS).

