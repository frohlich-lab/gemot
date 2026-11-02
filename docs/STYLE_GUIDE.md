# Coding standards

How code is written here. This is descriptive: it was derived from the existing ~53k lines,
so matching it means matching the code around you. For _what the system is_, see the
[README.md](../README.md). The server has its own
[style guide](https://github.com/frohlich-lab/PEtab-MCP/blob/main/docs/STYLE_GUIDE.md) — its error
hierarchy, where its constants and diagnostics live, how its tool docstrings are written.

## Rationale log

A non-obvious implementation choice is prepended to [RATIONALE.md](RATIONALE.md) and
references the source code where the implementation lives. The explanation names **the
failure that motivated it** — with the issue or PR number, the sweep date, or the measured
cost, whichever exists.

**Writing one is the exception.** Most PRs add nothing; the budget when they do is one entry
of two or three sentences, and a "why" that belongs to one module that still exists goes in that
module's docstring instead. See the header of [RATIONALE.md](RATIONALE.md) before adding one.

Two rules follow:

- **Negative knowledge is load-bearing.** "Deliberately NOT x", "do not add this back", "the
  first version of this matched … which caught …" — these are the record of what was already
  tried.
- **Write one when you make a call a reader would otherwise undo.** If the obvious
  simplification is wrong, the rationale log is the only thing standing between it and the
  next contributor.

This log explains _why_. It does not restate what the source code does.

## Docstrings

- **Every module opens with a substantial docstring**: what it is, what it owns, and why it
  is a separate module. For example, [ablations.py](../src/gemot/ablations.py).
- **Prose, not numpydoc.** A `Parameters ----------` block is the rare exception; explain in
  sentences.
- **Sphinx roles and double backticks**: `:func:`, `:mod:`, `:class:`, ` ``literal`` `.
- **`#:` for documented module-level constants**, so the comment attaches to the name.

## Agent-facing prose

Every string the harness sends to a model lives in
[prompts.py](../src/gemot/prompts.py), and the panel reads several of those surfaces in
one turn — its system prompt, each tool's description, whatever the finish gate returns. Three
rules keep one fact from being written down twice:

- **A rule read on more than one surface is ONE constant**, at the top of `prompts.py`, composed
  into every prompt that states it (`_SIGMA_CAP_ADVISORY`, `_MERGE_RULE`,
  `_RESIDUALS_DO_NOT_DECIDE_DONE`, `_DSCORE_SIGN`, `_PARAMETER_HALF`, `_MODEL_EDIT_OPS`). Never re-worded
  per surface: two paraphrases of one rule are two things to keep in sync, and the one the reader
  acts on is whichever they happened to read. A rule with one home is written where it lives.
- **No argument lists.** The JSON schema is the signature and is sent alongside the description,
  so `Args: a, b, c` is a hand-maintained copy of it. Say what an argument MEANS when the name
  does not; never enumerate them.
- **Derive a list of names, don't retype it.** `LEAD_SELF_READ_TOOLS` is rendered into the lead
  prompt AND spliced into `ORCHESTRATOR_TOOLS`, so the prompt cannot name a tool the lead does
  not hold.

The server's tool descriptions are the other half of this, and the panel reads them as the
server states them (frohlich-lab/PEtab-MCP#40): a tool's contract, and what an agent needs to use
it, is the server's to write, and a rule only this harness's job adds (its finish gate, its
analysts, when to merge) is the prompts'. Only the biology-blind arm replaces them.

## Types

- **`from __future__ import annotations` at the top of every module** — every module in
  `src/` bar one trivial `__init__.py`, and all but five test files.
- Builtin generics and `X | None`; `requires-python = ">=3.12"`, so there is no reason to
  reach for `typing.List` or `Optional`.
- **Return annotations on public functions** (~624 of 816 `def`s). Private one-liners may
  skip.

## Where things live

- **[prompts.py](../src/gemot/prompts.py)** — every agent-facing string the harness
  sends to a model, plus the registry (`PROMPT_GROUPS`) that versions them. Editing the text
  starts a new `prompt_version`, so results either side are not comparable.
- **[ablations.py](../src/gemot/ablations.py)** — every switch that changes what a
  run _does_ (what the agent is told, what tools it has access to). Switched on or off via
  environment variables.

## Dependencies

- **Every pin carries a why-comment** naming what breaks outside the range. `mcp>=2.3,<3`
  names what the 2.0 client API changed for the code here, which is what a 3.0 would break
  again.
- **An upper bound needs a "revisit once…" condition**, so the next person knows what would
  let them lift it.
- **A new import means a new pin.** Some dependencies are pinned precisely because
  nothing imports them directly (the server's `h5py` is written _through_ pyPESTO's storage
  API) — a transitive graph change would otherwise break them silently.
- **Optional deps must not silently skip tests.** `importorskip`-guarded tests are why `dev`
  depends on `gemot[agent]`. Adding an `importorskip` means adding its dependency to
  `dev`.

## Tests

- `tests/test_<module>.py`; plain `pytest` functions, no class wrappers.
- Short aliased imports of the module under test: `from gemot import analysis as A`.
- Module-level `_BASE_*` template strings with a `{extra}` splice point, plus a small
  `_helper` that fills it, rather than a fixture per variation
  (`_TOY_BASE_MODEL` and `reauthor_toy` in [conftest.py](../tests/conftest.py)).
- **A test comment names the regression it locks in.** A test whose failure does not tell
  you what broke is half a test — see the docstrings in
  [test_prompt_tracking.py](../tests/test_prompt_tracking.py).
- **Do not guard prose against prose.** A regex asserting that N paraphrases of one rule still
  agree is a maintenance cost standing in for a structural fix: make it one string and interpolate
  it (see "Agent-facing prose" above).
- A test that exercises an arm resolved at import time needs a subprocess, not
  monkeypatching ([test_subagent_registry.py](../tests/test_subagent_registry.py)).
- **`pytest` takes no `-n` from config.** The caller passes it; CI passes `-n 2`.

## Linting

`ruff check .` and `ruff format .` — config and the reason for every entry in `[tool.ruff]`
in [pyproject.toml](../pyproject.toml). CI runs `ruff check` as its own fast job (ruff only,
no toolchain).

The rule set is `E`, `F`, `W`, minus `E501`. It is narrow on purpose: it was
configured onto a codebase that had never run a linter, so it is what the code already
satisfies.

When a ruff upgrade lands a rule that is wrong for this repo, add it to `ignore` **with the
reason** — do not cap ruff.

## Commits

`area: what changed (#PR)` — lowercase area, imperative, matching `git log`. The area is a
module or subsystem (`fit:`, `judge:`, `finish_gate:`, `deps:`, `prompt:`) and may be
omitted for a change that spans the repo.
