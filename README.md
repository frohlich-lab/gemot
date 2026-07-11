# petab-mcp

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem so an
LLM/agent can **iteratively edit the model and its parameters and score the fit
against fixed data** — the loop behind model-building assistants such as
[petabgpt](https://github.com/frohlich-lab/petabgpt).

The experimental design is **fixed**: measurements, conditions and observables are
read-only. Only the **model** (SBML, edited via Antimony or SBML text) and the
**parameter table** are mutable. Simulation and objective evaluation use
[AMICI](https://github.com/AMICI-dev/AMICI); optimization uses
[pyPESTO](https://github.com/ICB-DCM/pyPESTO).

## Install

```bash
pip install -e '.[amici,optimize]'      # full: simulation + optimization
pip install -e '.[amici]'               # simulation/objective only
pip install -e .                        # read/edit/validate only (no AMICI)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja,
and the pip `swig` binary on `PATH` (OpenBLAS ships inside the AMICI wheel). The
server auto-adds the `swig` package's bundled binary to `PATH` if it isn't found.

## Run

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# or: PETAB_MCP_YAML=path/to/problem.yaml petab-mcp
```

This starts an stdio MCP server. Point an MCP client at it, e.g.:

```json
{
  "mcpServers": {
    "petab": {
      "command": "petab-mcp",
      "args": ["--petab-yaml", "/abs/path/to/problem.yaml"]
    }
  }
}
```

A ready-to-use toy problem (reversible conversion reaction A⇌B) lives at
`tests/data/toy_v1/problem.yaml`.

## Tools

**Read-only (the fixed design)**
- `describe_problem` — model size, tables, free parameters, AMICI availability
- `list_observables`, `list_conditions`
- `get_measurements(condition_id?, observable_id?)`

**Model (mutable)**
- `get_model(format="antimony"|"sbml")`
- `update_model(text, format)` — validated against the fixed design; an edit that
  breaks a referenced observable/condition/parameter is rejected
- `get_model_structure` — species / parameters / reactions
- `set_model_parameter(parameter_id, value)` — in-place SBML edit (minimal, extensible)
- `validate_problem` — lint the current problem

**Parameters (mutable)**
- `get_parameters`
- `update_parameters(updates)` — transactional edits of
  `nominalValue / lowerBound / upperBound / estimate / parameterScale`
- `sync_parameters(policy="report"|"add")` — reconcile the table with the model

**Evaluation**
- `simulate` — simulated vs measured values per data point
- `evaluate_objective` — negative log-likelihood (+ χ²/residuals when available)
- `optimize(n_starts, seed)` — pyPESTO multi-start; writes the best (unscaled) fit back

**Session**
- `reset` — revert model/parameter edits to the on-disk problem

Read-only resources mirror the tables: `petab://model/antimony`, `petab://model/sbml`,
`petab://observables`, `petab://conditions`, `petab://measurements`,
`petab://parameters`.

## Design notes

- **The fixed invariant is structural**: there is simply no tool that writes the
  measurement/condition/observable tables, and every model edit is re-linted with
  petab's own checks before it is committed — an edit that drops a referenced entity
  is rejected.
- **Cheap parameter edits, cached compilation**: AMICI-compiled models are cached by
  SBML hash. A model-structure edit recompiles under a new hash; parameter edits do
  not — the current `nominalValue`s are passed to the simulator as `problem_parameters`.
- **Optimization**: pyPESTO builds an AMICI-backed objective from the v1 problem and
  runs multi-start optimization; the best parameter vector is unscaled per
  `parameterScale` before being written back into `nominalValue`.

## Development

```bash
pip install -e '.[amici,optimize,dev]'
pytest            # AMICI/pyPESTO-dependent tests skip automatically if absent
```

## License

MIT
