# CLI drivers

`--driver codex` runs one `codex exec` process; `--driver claude` runs one `claude -p`
process for the entire task. OpenRouter remains the default driver. Install
`pip install -e '.[harness]'` and a current official CLI, then authenticate with `codex login`
or `claude auth login` outside gemot.

```bash
python examples/predator_prey/run.py --driver codex --model YOUR_CODEX_MODEL_ID
python examples/predator_prey/run.py --driver claude --model sonnet
```

`--model` is required and passed directly to the CLI. Gemot leaves authentication to that
CLI: it neither extracts credentials nor falls back to another provider after an error.
`--api-key` configures OpenRouter and is rejected for these drivers.

## Native execution

Gemot supplies `scenario.agent_task` once, with a short static introduction to the MCP tools.
The CLI owns subsequent reasoning, tool calls, conversation history, and compaction. There
is no LangChain chat-model shim, serialized turn-by-turn prompt, imposed panel, finish gate,
forced delegation, or output schema. Native transcript management preserves the CLI's
prompt-cache opportunities; it cannot guarantee a cache hit or a particular hit rate.

A localhost MCP gateway exposes the shared scored PEtab tools, including `make_changes`.
Heldout judging tools are excluded. Built-in file/shell access and user customizations are
restricted to preserve benchmark blindness. This initial implementation provides only PEtab
MCP, without external biology MCP sessions. The existing judge evaluates the resulting model
after the agent session ends.

Claude's tools are loaded before the initial prompt. The gateway requests its supported
500,000-character inline tool-result maximum; larger replies can still spill to a file that
this restricted session cannot read. Use filtered data queries for such responses. See
[Claude's output limits](https://code.claude.com/docs/en/mcp#raise-the-limit-for-a-specific-tool).

| Option | Meaning |
| --- | --- |
| `--codex-executable`, `--claude-executable` | CLI path; defaults to `codex` / `claude` |
| `--codex-timeout`, `--claude-timeout` | Optional seconds for the entire session; no default cap |
| `--max-wall-seconds` | Overall run budget; when both limits exist, the earlier deadline applies |
| `--reasoning-effort` | Native model setting; omitted or `default` preserves the CLI default |

Codex accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; Claude accepts `low`,
`medium`, `high`, `xhigh`, `max`, subject to model support. The CLI owns context limits.
`--biocurator-model` is rejected; temperature and recursion limit are inapplicable and recorded
as null. Preflight checks required CLI features before recreating `work/`. Results record the
executable version and settings. Cancellation and timeouts terminate and reap the process.

## Phoenix observations

Run `phoenix serve` and set `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`, or let gemot
start a local collector; `--phoenix-keep-alive` retains that local UI after completion.
Missing tracing packages produce a warning. Results link `phoenix_trace_id` and
`phoenix_project`.

The observer reads native JSONL events independently of the agent architecture. Phoenix
receives session, observed model-output, and native tool spans alongside actual gateway tool
spans. Raw events are archived as `work/codex_events.jsonl` or `work/claude_events.jsonl`.
Codex exposes output items and aggregate usage, insufficient to reconstruct every underlying
API call. Claude exposes response boundaries and usage. Timings reflect observed events;
hidden prompts and per-call token counts are not invented. Cache reads/creation are recorded
when reported. Claude's reported dollar cost is an estimate; billed cost remains unknown.
Traces and logs can contain scientific prompts and tool results.

The upstream Arize [Codex observability skill](https://github.com/Arize-ai/coding-harness-tracing/blob/main/tracing/codex/skills/manage-codex-tracing/SKILL.md)
illustrates independent native tracing. Gemot does not install its hooks or change global
CLI configuration; its own event observer has the visibility limits above.

## ChatGPT terms assessment

Reviewed 26 October 2026; this is an implementation assessment, not legal advice.

OpenAI documents [`codex exec` automation](https://learn.chatgpt.com/docs/non-interactive-mode)
and [ChatGPT/API-key authentication](https://learn.chatgpt.com/docs/auth). Official CLI use for
scientific work therefore has a documented basis, but that does not establish blanket
permission for subscription-funded bulk benchmarks.

The [consumer terms](https://openai.com/policies/terms-of-use/) and
[Europe terms](https://openai.com/policies/eu-terms-of-use/) restrict automated extraction,
credential sharing, circumventing limits or safeguards, and developing competing models from
outputs. Scientific mechanistic model fitting differs from developing a competing AI model;
the campaign's purpose and scale still matter.

For sustained unattended benchmarks or CI, prefer separately billed API-key authentication,
OpenAI's default automation recommendation, under the applicable
[Services Agreement](https://openai.com/policies/services-agreement/). API use retains policy
and contractual restrictions. Seek written OpenAI clarification for a large campaign funded
by a ChatGPT subscription. Claude's [automation documentation](https://code.claude.com/docs/en/headless)
describes its interface; Anthropic's separate terms require their own assessment.
