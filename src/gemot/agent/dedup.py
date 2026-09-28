"""Identity and de-duplication for proposed tool calls.

Pure functions over plain dicts — no langchain, no third-party import at all — so the guard logic
in :mod:`.guards` that uses them is unit-testable on its own.
"""

from __future__ import annotations


def _tool_call_key(tc: dict) -> str:
    """A stable identity for a proposed tool call: its name + canonicalised arguments.

    Two calls with the same name and the same arguments are the same request regardless of the
    id the provider stamped on them, so they compare equal here.
    """
    import json

    args = tc.get("args")
    try:
        args_norm = json.dumps(args, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - non-serialisable args fall back to repr
        args_norm = repr(args)
    return f"{tc.get('name')}\x00{args_norm}"


def _dedupe_tool_calls(tool_calls: "list[dict] | None") -> "tuple[list[dict], int]":
    """Drop exact-duplicate (same name+args) tool calls, keeping the FIRST of each.

    Accepts ``None`` (treated as empty). Returns ``(kept, n_dropped)``. Pure — no langchain
    types — so it is unit-testable without the agent stack.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    for tc in tool_calls or []:
        key = _tool_call_key(tc)
        if key in seen:
            continue
        seen.add(key)
        kept.append(tc)
    return kept, len(tool_calls or []) - len(kept)


def _dedupe_raw_tool_calls(raw: "list[dict] | None") -> list[dict]:
    """De-dupe the OpenAI-format raw tool-call list from ``additional_kwargs``.

    Mirrors :func:`_dedupe_tool_calls` but on the provider's raw shape
    (``{"id", "type", "function": {"name", "arguments"}}``), keying on name + the raw arguments
    STRING and keeping the first of each — so it collapses a repeated call regardless of whether
    the provider reused its ``id`` (matching by id membership alone would keep both copies of a
    repeated-id duplicate).
    """
    seen: set[tuple] = set()
    kept: list[dict] = []
    for c in raw or []:
        fn = c.get("function") or {}
        key = (fn.get("name"), fn.get("arguments"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept
