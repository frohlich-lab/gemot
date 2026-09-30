# Working in this repo

**`gemot`** (under `src/`) is the agent harness that drives the petab-mcp server: a deepagents
panel, the judge, the sweep analysis, the run explorer and the paper pipeline.

Two neighbours pin what they depend on by commit:
- the server, one PEtab v1 problem behind a _deliberately constrained_ editing loop whose
  constraints are the product, lives in [frohlich-lab/PEtab-MCP](https://github.com/frohlich-lab/PEtab-MCP),
  pinned here in `pyproject.toml`;
- the benchmark gemot drives (`examples/`, `slurm/`, `paper/`) lives in
  [frohlich-lab/petab_mcp](https://github.com/frohlich-lab/petab_mcp), which pins gemot.
A change here reaches the benchmark when its gemot pin is bumped.

## Read before you change behaviour

The "Overview" and "Design Notes" sections of the
[PEtab MCP README](https://github.com/frohlich-lab/PEtab-MCP#design-notes). The design principles
of the package are defined there.

## Writing code here

Refer to [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md).
