"""A failure of the server's infrastructure is not the model's problem, so it ends the run.

The server builds, fits and stores every model on a filesystem the agent can neither see nor fix.
When that filesystem fills, the writes fail -- ``[Errno 122] Disk quota exceeded`` (EDQUOT) on a
quota, ``[Errno 28] No space left on device`` (ENOSPC) on a full disk -- and the server used to hand
the failure to the agent as if it were a modelling problem. In the INDRA-arm sweep that motivated
this module (gemot#39, PEtab-MCP#83) the lab quota filled for a few hours: 241 ``make_changes``
calls came back "could not be compiled/fit ... ill-posed", 40 more carried the raw errno, 15
servers segfaulted storing a result, and the driver kept every run going. Leads re-authored models
that were fine until the dedup guard refused them, one finished early because "the search is
limited by those compile failures", and 30 runs that completed carried the damage with no mark in
result.json.

This module is the ONE definition of what such a failure looks like and of the stop it causes.
There are two kinds of evidence (:func:`find`):

* the server's own report (PEtab-MCP#95): a tool error whose text starts
  ``infrastructure_error: <cause>: <detail>``, ``disk_full`` being the only cause so far;
* the raw disk-full text, from a server that predates that report, or anywhere the report does not
  reach: an ``OSError``, a linker's ``final link failed: Disk quota exceeded``, HDF5's error stack.

It is read in four places:

* the tool wrapper (:mod:`gemot.agent.circuit_breaker`) reads every core tool's result and error,
  and raises :class:`InfrastructureError` in place of the result;
* :class:`LogWatch` reads the server's own logs. Before PEtab-MCP#95 they are the only place an
  "ill-posed" rejection shows its real cause, and they are the only trace a segfaulted server
  leaves;
* the driver ends the run on either, with no resume, and the runner skips the judge;
* the analysis files :func:`stop_reason` in its own bucket, outside every success rate, so the
  sweep can resubmit the run.

Stdlib only: the agent package imports this (tests/test_import_lightness.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

#: The ``stop_reason`` prefix of every run the infrastructure failed under, and of the server's
#: report (``petab_mcp.errors.INFRASTRUCTURE_ERROR``; the two packages cannot share a constant).
STOP_PREFIX = "infrastructure_error"

#: The one cause so far (``petab_mcp.errors.DISK_FULL``).
DISK_FULL = "disk_full"


def stop_reason(cause: str) -> str:
    """The ``stop_reason`` of a run stopped by ``cause``. A bare label, like
    ``wall_time_budget_exceeded``: the evidence goes in ``infrastructure_evidence``, so the label
    never carries numbers or paths for a substring marker to trip on."""
    return f"{STOP_PREFIX}: {cause}"


#: The stop reason a full disk writes.
STOP_REASON = stop_reason(DISK_FULL)

# The server's report, wherever it lands in a message: alone, or behind the exception's own name.
_SERVER_REPORT = re.compile(rf"\b{STOP_PREFIX}: ([a-z_]+): ")

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

_CONTEXT = 120  # characters of evidence kept either side of a raw match, or after a report's start


class Evidence(NamedTuple):
    """What failed (``cause``) and the text that says so, on one line."""

    cause: str
    text: str


def find(text: str) -> Evidence | None:
    """The evidence of an infrastructure failure in ``text``, or ``None``.

    The server's report wins, since it names its cause; otherwise the raw disk-full text."""
    if (report := _SERVER_REPORT.search(text)) is not None:
        return Evidence(report.group(1), _one_line(text[report.start() : report.end() + _CONTEXT]))
    if (match := _DISK_FULL.search(text)) is not None:
        start, end = max(0, match.start() - _CONTEXT), match.end() + _CONTEXT
        return Evidence(DISK_FULL, _one_line(text[start:end]))
    return None


def find_in_error(exc: BaseException) -> Evidence | None:
    """:func:`find` over an exception: its own text, every leaf of an exception group (the anyio
    task groups around the MCP session wrap the real error in one), and what it was raised
    from. An :class:`InfrastructureError` gives its own evidence, which names the tool."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, InfrastructureError):
            return e.evidence
        if (hit := find(f"{type(e).__name__}: {e}")) is not None:
            return hit
        if isinstance(e, BaseExceptionGroup):
            stack.extend(e.exceptions)
        stack.extend((e.__cause__, e.__context__))
    return None


def is_stop(stop_reason: object) -> bool:
    """Whether ``stop_reason`` is an infrastructure stop (:data:`STOP_PREFIX`)."""
    return str(stop_reason or "").strip().lower().startswith(STOP_PREFIX)


def _one_line(text: str) -> str:
    return " ".join(text.split())


class InfrastructureError(Exception):
    """Raised by the tool wrapper in place of a result that met an infrastructure failure.

    Deliberately a plain ``Exception``, not a ``ToolException``: a tool's ``handle_tool_error``
    turns the adapter's ``ToolException`` into a result the model is asked to work around -- which
    is exactly where the server's own report would otherwise land -- while langgraph's tool node
    re-raises anything but an argument error. So this ends the super-step and reaches the driver,
    which ends the run.
    """

    def __init__(self, evidence: Evidence) -> None:
        super().__init__(evidence.text)
        self.evidence = evidence


class LogWatch:
    """The infrastructure failures the server's logs gained since the last look.

    Called with no arguments (it is the driver's ``infrastructure_probe``) it reads each log from
    where it stopped, adds the lines carrying evidence to :attr:`lines`, and returns the first new
    :class:`Evidence`, or ``None``. A missing or unreadable log reads as nothing, so the probe
    never ends a run by accident. It reads whole lines only, so a line still being written is read
    once it is complete; ``final=True`` takes the rest too, since a dead server's last line has no
    newline.
    """

    def __init__(self, *paths: Path) -> None:
        self.paths = paths
        self.lines = 0
        self.evidence: Evidence | None = None  # the first found, over the whole run
        self._offsets = dict.fromkeys(paths, 0)

    def __call__(self, final: bool = False) -> Evidence | None:
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
                if (hit := find(line)) is not None:
                    self.lines += 1
                    new = new or Evidence(hit.cause, f"{path.name}: {hit.text}")
        self.evidence = self.evidence or new
        return new
