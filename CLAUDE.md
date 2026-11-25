# Working in this repo

**`gemot`** (under `src/`) is the agent harness that drives the petab-mcp server: a deepagents
panel, the judge, the sweep analysis and the run explorer.

Its neighbours pin what they depend on by commit:
- the server, one PEtab v1 problem behind a _deliberately constrained_ editing loop whose
  constraints are the product, lives in [frohlich-lab/PEtab-MCP](https://github.com/frohlich-lab/PEtab-MCP)
  and is released to PyPI as `petab-mcp`, pinned here in `pyproject.toml` by release;
- the problems gemot drives, with the code that materializes them and fits their reference bars,
  live in [frohlich-lab/PEtabAgentBench](https://github.com/frohlich-lab/PEtabAgentBench), whose
  `gemot` extra pins gemot;
- the study (`examples/` mounting PEtabAgentBench, `slurm/`, `paper/`) lives in
  [frohlich-lab/gemot_iclr_2026](https://github.com/frohlich-lab/gemot_iclr_2026), which pins gemot.
A change here reaches the benchmark when its gemot pin is bumped.

## Read before you change behaviour

The "Overview" section of the [PEtab MCP README](https://github.com/frohlich-lab/PEtab-MCP) and
its [design notes](https://github.com/frohlich-lab/PEtab-MCP/blob/main/docs/DESIGN.md). The design
principles of the package are defined there.

## Writing code here

Refer to [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md).
