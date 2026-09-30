# Working in this repo

**`petab_harness`** (under `src/`) is the agent harness that drives the petab-mcp server in
`examples/`: a deepagents panel, the judge, the benchmark sweep, and the run explorer.

The server itself — one PEtab v1 problem behind a _deliberately constrained_ editing loop, whose
constraints are the product, not scaffolding — lives in
[frohlich-lab/PEtab-MCP](https://github.com/frohlich-lab/PEtab-MCP) and is installed pinned to a
commit (`pyproject.toml`). A server change lands there first; bumping the pin brings it here.

## Read before you change behaviour

The "Overview" and "Design Notes" sections of the
[PEtab MCP README](https://github.com/frohlich-lab/PEtab-MCP#design-notes). The design principles
of the package are defined there.

## Writing code here

Refer to [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md).
