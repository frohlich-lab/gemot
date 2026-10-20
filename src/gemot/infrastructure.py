"""A full disk under the server is not the model's problem, so it ends the run.

The server builds, fits and stores every model on a filesystem the agent can neither see nor fix.
When that filesystem fills, the writes fail -- ``[Errno 122] Disk quota exceeded`` (EDQUOT) on a
quota, ``[Errno 28] No space left on device`` (ENOSPC) on a full disk -- and the server hands the
failure to the agent as if it were a modelling problem. In the INDRA-arm sweep that motivated this
module (gemot#39, PEtab-MCP#83) the lab quota filled for a few hours: 241 ``make_changes`` calls
came back "could not be compiled/fit ... ill-posed", 40 more carried the raw errno, 15 servers
segfaulted storing a result, and the driver kept every run going. Leads re-authored models that
were fine until the dedup guard refused them, one finished early because "the search is limited by
those compile failures", and 30 runs that completed carried the damage with no mark in result.json.

This module is the ONE definition of that signature and of the stop it causes:

* the tool wrapper (:mod:`gemot.agent.circuit_breaker`) matches it in every core tool's result and
  error, and raises :class:`InfrastructureError` in place of the result;
* :class:`LogWatch` matches it in the server's own logs. On the current server they are the only
  place an "ill-posed" rejection shows its real cause (``ld: final link failed: Disk quota
  exceeded`` in ``amici_build.log``), and the only trace a segfaulted server leaves;
* the driver ends the run on either, with no resume, and the runner skips the judge;
* the analysis files :data:`STOP_REASON` in its own bucket, outside every success rate, so the
  sweep can resubmit the run.

The server's half (PEtab-MCP#83) is to raise a dedicated, non-retryable error instead of the
"ill-posed" message. Until the pin reaches it, the raw errno text is what there is to match. Stdlib
only: the agent package imports this (tests/test_import_lightness.py).
"""

from __future__ import annotations

import re
from pathlib import Path

#: The ``stop_reason`` prefix of every run the infrastructure failed under. A bare label, like
#: ``wall_time_budget_exceeded``: the evidence goes in ``disk_full_evidence``, so the label never
#: carries numbers or paths for a substring marker to trip on.
STOP_PREFIX = "infrastructure_error"

#: The stop reason a full disk writes.
STOP_REASON = f"{STOP_PREFIX}: disk_full"

# A failed write, wherever it is printed: Python's OSError (`[Errno 28] No space left on device`),
# the strerror text alone (a linker's `final link failed: Disk quota exceeded`, `cmake --install`),
# HDF5's error stack (`errno = 122, error message = 'Disk quota exceeded'`), and the errno names.
# 28 and 122 are Linux's ENOSPC and EDQUOT, the platform the sweeps run on; the strerror text is
# the same everywhere.
_DISK_FULL = re.compile(
    r"\[Errno (?:28|122)\]"
    r"|\berrno\s*=\s*(?:28|122)\b"
    r"|No space left on device"
    r"|Disk quota exceeded"
    r"|\bE(?:NOSPC|DQUOT)\b"
)

_CONTEXT = 120  # characters kept either side of a match as evidence


class InfrastructureError(Exception):
    """Raised by the tool wrapper in place of a result that met a full disk.

    Deliberately a plain ``Exception``, not a ``ToolException``: a tool's ``handle_tool_error``
    may turn a ``ToolException`` into a result the model is asked to work around, while langgraph's
    tool node re-raises anything but an argument error. So this ends the super-step and reaches the
    driver, which ends the run.
    """


def disk_full(text: str) -> str | None:
    """The stretch of ``text`` around its first disk-full signature, on one line; ``None`` if
    there is none."""
    match = _DISK_FULL.search(text)
    if match is None:
        return None
    start, end = max(0, match.start() - _CONTEXT), match.end() + _CONTEXT
    return " ".join(text[start:end].split())


def disk_full_error(exc: BaseException) -> str | None:
    """:func:`disk_full` over an exception: its own text, every leaf of an exception group (the
    anyio task groups around the MCP session wrap the real error in one), and what it was raised
    from."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if (hit := disk_full(f"{type(e).__name__}: {e}")) is not None:
            return hit
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        stack.extend((e.__cause__, e.__context__))
    return None


def is_stop(stop_reason: object) -> bool:
    """Whether ``stop_reason`` is an infrastructure stop (:data:`STOP_PREFIX`)."""
    return str(stop_reason or "").strip().lower().startswith(STOP_PREFIX)


class LogWatch:
    """The disk-full lines the server's logs gained since the last look.

    Called with no arguments (it is the driver's ``disk_full_probe``) it reads each log from where
    it stopped, adds the lines carrying the signature to :attr:`lines`, and returns the first new
    one, or ``None``. A missing or unreadable log reads as nothing, so the probe never ends a run
    by accident. It reads whole lines only, so a line still being written is read once it is
    complete; ``final=True`` takes the rest too, since a dead server's last line has no newline.
    """

    def __init__(self, *paths: Path) -> None:
        self.paths = paths
        self.lines = 0
        self.evidence: str | None = None  # the first line found, over the whole run
        self._offsets = dict.fromkeys(paths, 0)

    def __call__(self, final: bool = False) -> str | None:
        new = None
        for path in self.paths:
            try:
                with path.open("rb") as fh:
                    fh.seek(self._offsets[path])
                    chunk = fh.read()
            except OSError:
                continue
            end = len(chunk) if final else chunk.rfind(b"\n") + 1
            self._offsets[path] += end
            for line in chunk[:end].decode("utf-8", "replace").splitlines():
                if (hit := disk_full(line)) is not None:
                    self.lines += 1
                    new = new or f"{path.name}: {hit}"
        self.evidence = self.evidence or new
        return new
