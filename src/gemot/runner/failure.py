"""Ending a run honestly when it aborts before a verdict.

A provider or connection failure must not vanish: the run writes a DEGRADED ``result.json``
recording what happened, so a sweep can tell an infrastructure abort apart from a modeling
failure instead of scoring an empty model as a loss.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .provenance import _now_iso
from ..scenario import Scenario

def _summarize_exc(exc: BaseException) -> str:
    """A short, readable one-liner for an exception — unwrapping an ExceptionGroup (anyio's
    TaskGroup wraps a background provider/connection failure in one) down to its first leaf,
    so 'ExceptionGroup: unhandled errors in a TaskGroup' becomes e.g. 'ReadTimeout: ...'.

    ``BaseExceptionGroup`` is a builtin (Python >= 3.11, well under the project's 3.12 floor)."""
    seen: BaseException = exc
    while isinstance(seen, BaseExceptionGroup) and seen.exceptions:
        seen = seen.exceptions[0]
    return f"{type(seen).__name__}: {seen}"[:200]


def _should_reraise(exc: BaseException) -> bool:
    """Whether a broadened ``except BaseException`` guard must let ``exc`` propagate rather
    than turn it into a degraded result.

    The degraded-result guards are widened from ``except Exception`` to ``except
    BaseException`` so a ``BaseExceptionGroup`` from anyio's TaskGroup teardown — whose leaf
    is a *background* ``asyncio.CancelledError`` raised while the
    streamablehttp_client/ClientSession task group unwinds after a provider stall — is
    caught. Such a group is a ``BaseException`` but NOT an ``Exception`` (an ``ExceptionGroup``
    of all-``Exception`` leaves *is* an ``Exception`` and was already caught), so the original
    guards let it escape and vanish the run with no result.json.

    Three things must still propagate untouched:

    * ``KeyboardInterrupt`` / ``SystemExit`` — user interrupt and interpreter exit (the
      latter is how a missing API key hard-stops the run), including a group that carries one
      as a leaf.
    * a *bare* ``asyncio.CancelledError`` — a genuine top-level cancellation of THIS
      coroutine; swallowing it would break cooperative cancellation. (A ``CancelledError``
      *wrapped in a group* is the anyio-teardown case above and is NOT re-raised.)
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return exc.subgroup((KeyboardInterrupt, SystemExit)) is not None
    return False


def _write_degraded_result(work: Path, base_meta: dict, scenario: Scenario,
                           reason: str, server_returncode: int | None = None,
                           clock=None) -> dict:
    """Persist (and return) the degraded ``result.json`` written when a run aborts before a
    verdict, so the batch and the cross-run aggregator find a well-formed row instead of a
    missing file.

    Keeps the contract ``collect_metrics.py`` buckets on: ``scenario``, ``stop_reason``,
    ``success`` (False), ``delta_bic`` (None), ``run_ended_at``, plus everything in
    ``base_meta`` (driver/model/version/git_sha/run_started_at/phoenix ids/...).

    ``server_returncode`` (when known) records whether the MCP server subprocess had already
    exited when the run aborted — the single most useful signal for diagnosing a transport
    ``ReadError``: a negative value is a fatal SIGNAL (e.g. ``-11`` = SIGSEGV, a hard crash
    during a tool call such as an AMICI import), distinguishing "the server died" from a mere
    network blip (server still alive, ``None``)."""
    server_note = ""
    if server_returncode is not None:
        if server_returncode < 0:
            server_note = f" [MCP server was killed by signal {-server_returncode}]"
        elif server_returncode > 0:
            server_note = f" [MCP server had exited with code {server_returncode}]"
        else:
            server_note = " [MCP server had exited cleanly]"
    print(f"[run] aborted by an unrecoverable error ({reason}){server_note}; "
          "writing a degraded result.json and exiting.")
    result = {
        **base_meta,
        "scenario": scenario.name,
        "stop_reason": f"aborted: {reason}",
        "success": False,
        "delta_bic": None,
        "run_ended_at": _now_iso(),
        "server_returncode": server_returncode,
        "error": "run aborted before a verdict (provider/connection failure)",
        # An aborted run still made edits, and they are the MOST interesting ones: omitting them
        # scored every abort as zero rejections, biasing the sweep-wide rate DOWN for exactly the
        # runs that struggled. `clock` is None only when the abort predates the session.
        **({"tool_calls": clock.calls, "tool_call_counts": dict(clock.by_tool),
           "edit_rejects": clock.reject_summary()} if clock else {}),
    }
    (work / "result.json").write_text(json.dumps(result, indent=2))
    return result
