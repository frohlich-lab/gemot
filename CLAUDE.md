# Working in this repo

Two packages under `src/`:

- **`petab_mcp`** — the MCP server. It wraps one PEtab v1 parameter-estimation problem and
  exposes a _deliberately constrained_ editing loop. The constraints are the product, not
  scaffolding.
- **`petab_harness`** — the agent harness that drives that server in `examples/`: a
  deepagents panel, the judge, the benchmark sweep, and the run explorer.

## Read before you change behaviour

The "Overview" and "Design Notes" sections of [README.md](README.md). The design principles
of the package are defined there.

## Writing code here

Refer to [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md).
